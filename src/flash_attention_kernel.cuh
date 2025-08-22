template <typename T>
__global__ void flashAttentionKernel(
    T* d_q, T* d_k, T* d_v, T* d_o,
    int target_seq_len, int src_seq_len,
    int head_dim, int group,
    float softmax_scale, bool is_causal,
    int B_r, int B_c, int T_r, int T_c,
    int qo_stride_b, int qo_stride_s, int qo_stride_n,
    int kv_stride_b, int kv_stride_s, int kv_stride_n) {
    
    size_t bx = blockIdx.x;  // batch                          -> batch_size
    size_t by = blockIdx.y;  // q's head index                 -> query_head
    size_t tx = threadIdx.x; // q's row index within one block -> B_r/B_c

    size_t qo_offset = bx * qo_stride_b + by * qo_stride_n;
    size_t kv_offset = bx * kv_stride_b + by / group * kv_stride_n;

    extern __shared__ float shared_mem[];
    T *q_i = reinterpret_cast<T *>(shared_mem);
    T *k_j = reinterpret_cast<T *>(q_i + B_r * head_dim);
    T *v_j = reinterpret_cast<T *>(k_j + B_c * head_dim);
    T *s_i = reinterpret_cast<T *>(v_j + B_c * head_dim);

    for (size_t i = 0; i < T_r; ++i) {
        // skip when over q's seq_len
        if (i * B_r + tx >= target_seq_len) {
            break;
        }

        // load q_i from HBM to on-chip SRAM
        for (size_t x = 0; x < head_dim; ++x) {
            q_i[tx * head_dim + x] = d_q[qo_offset + (i * B_r + tx) * qo_stride_s + x];
        }
        // initial m, l
        T row_m_prev = -INFINITY;
        T row_l_prev = 0;

        for (size_t j = 0; j < T_c; ++j) {
            __syncthreads();
            // load k_j, v_j from HBM to on-chip SRAM
            for (size_t y = 0; y < B_c; ++y) {
                for (size_t x = 0; x < head_dim; ++x) {
                    k_j[y * head_dim + x] = d_k[kv_offset + (y + j * B_c) * kv_stride_s + x];
                    v_j[y * head_dim + x] = d_v[kv_offset + (y + j * B_c) * kv_stride_s + x];
                }
            }
            // for (size_t x = 0; x < head_dim; ++x) {
            //     k_j[tx * head_dim + x] = d_k[kv_offset + (tx + j * B_c) * kv_stride_s + x];
            //     v_j[tx * head_dim + x] = d_v[kv_offset + (tx + j * B_c) * kv_stride_s + x];
            // }
            __syncthreads();

            T row_m = -INFINITY;
            for (size_t y = 0; y < B_c; ++y) {
                if (j * B_c + y >= src_seq_len) {
                    break;
                }

                // Causal mask
                if (is_causal && (i * B_r + tx < j * B_c + y)) break;

                // S_i^(j) = Q_i @ K_j^T / softmax_scale
                T sum = 0;
                for (size_t x = 0; x < head_dim; ++x) {
                    sum += q_i[tx * head_dim + x] * k_j[y * head_dim + x];
                }
                sum *= softmax_scale;
                s_i[tx * B_c + y] = sum;

                row_m = max(row_m, sum);
            }

            // m_i^(j) = max(m_i^(j - 1), rowmax(S_i^(j)))
            T new_row_m;
            new_row_m = max(row_m_prev, row_m);

            // rowsum(P_i^(j))
            T row_l = 0;
            for (size_t y = 0; y < B_r; ++y) {
                if (j * B_c + y >= src_seq_len) {
                    break;
                }

                // Causal mask
                if (is_causal && (i * B_r + tx < j * B_c + y)) break;

                // P_i^(j) = exp(S_i^(j) - m_i^(j))
                s_i[tx * B_c + y] = __expf(s_i[tx * B_c + y] - new_row_m);

                row_l += s_i[tx * B_c + y];
            }

            // l_i^(j) = exp(m_i^(j - 1) - m_i^(j - 1)) * l_i^(j - 1) + rowsum(P_i^(j))
            T row_m_exp = __expf(row_m_prev - new_row_m);
            T new_row_l = (row_m_exp * row_l_prev) + row_l;

            // out_i^(j) = diag(exp(m_i^(j - 1) - m_i^(y))) * O_i^(j - 1) + P_i^(j) * V_j
            for (size_t x = 0; x < head_dim; ++x) {
                T pv = 0;
                for (size_t y = 0; y < B_c; ++y) {
                    if (j * B_c + y >= src_seq_len) {
                        break;
                    }

                    // Causal mask
                    if (is_causal && (i * B_r + tx < j * B_c + y)) break;

                    pv += s_i[tx * B_c + y] * v_j[y * head_dim + x];
                }

                d_o[qo_offset + (i * B_r + tx) * qo_stride_s + x] = row_m_exp * d_o[qo_offset + (i * B_r + tx) * qo_stride_s + x] + pv;
            }

            row_m_prev = new_row_m;
            row_l_prev = new_row_l;
        }

        // O_i = O_i^(Tc) / l_i^(Tc)
        for (size_t x = 0; x < head_dim; ++x) {
            d_o[qo_offset + (i * B_r + tx) * qo_stride_s + x] /= row_l_prev;
        }
    }
}
