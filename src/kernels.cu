#include <vector>
#include <random>

#include "../tester/utils.h"
#include "kth_largest_kernel.cuh"
#include "flash_attention_kernel.cuh"


/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
    // 无效测例
    if (h_input.empty() || k == 0 || k > h_input.size()) {
        return T(-100);
    }

    size_t total_size = h_input.size();

    // 两个缓冲，交替读写
    T *d_curr = nullptr, 
      *d_next = nullptr;
    cudaMalloc(&d_curr, total_size * sizeof(T));
    cudaMalloc(&d_next, total_size * sizeof(T));
    cudaMemcpy(d_curr, h_input.data(), total_size * sizeof(T), cudaMemcpyHostToDevice);

    int *d_mark = nullptr, 
        *d_mark_equal = nullptr;
    cudaMalloc(&d_mark, total_size * sizeof(int));
    cudaMalloc(&d_mark_equal, total_size * sizeof(int));
    int *d_scan = nullptr,
        *d_scan_equal = nullptr;
    cudaMalloc(&d_scan, total_size * sizeof(int));
    cudaMalloc(&d_scan_equal, total_size * sizeof(int));

    std::default_random_engine rng;

    size_t curr_size = total_size;

    while (true) {
        std::uniform_int_distribution<size_t> dist(0, curr_size - 1);
        // 随机选一个点为 pivot
        T pivot;
        cudaMemcpy(&pivot, d_curr + dist(rng), sizeof(T), cudaMemcpyDeviceToHost);

        dim3 block(256);
        dim3 grid((curr_size + block.x - 1) / block.x);

        // 标记
        markKernel<<<grid, block>>>(d_curr, d_mark, d_mark_equal, curr_size, pivot);
        cudaDeviceSynchronize();

        // 求前缀和
        prefixSum(d_mark, d_scan, curr_size);
        prefixSum(d_mark_equal, d_scan_equal, curr_size);

        // numLeft : 数组中 >= pivot 的数量
        int numLeft = 0;
        cudaMemcpy(&numLeft, d_scan + curr_size - 1, sizeof(int), cudaMemcpyDeviceToHost);

        // numEqual : 数组中 == pivot 的数量
        int numEqual = 0;
        cudaMemcpy(&numEqual, d_scan_equal + curr_size - 1, sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // 依据 k 缩小到对应半边
        if (numLeft - numEqual < k && k <= numLeft) {
            cudaFree(d_curr);
            cudaFree(d_next);
            cudaFree(d_mark);
            cudaFree(d_scan);
            cudaFree(d_mark_equal);
            cudaFree(d_scan_equal);
            return pivot;
        } else if (numLeft < k) {
            // 目标在右半：(numLeft, end) —— 实际是 < pivot 的区间
            scatterAllKernel<<<grid, block>>>(d_curr, d_next, d_mark, d_scan, curr_size, false);
            std::swap(d_curr, d_next);
            curr_size -= numLeft;
            k -= numLeft;
        } else {
            // 目标在左半：[0, numLeft)
            scatterAllKernel<<<grid, block>>>(d_curr, d_next, d_mark, d_scan, curr_size, true);
            std::swap(d_curr, d_next);
            curr_size = numLeft;
        }
    }

    return T(-1000);
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    int B_r = 16;
    int B_c = 16;
    
    int T_r = ceil(float(target_seq_len) / B_r);
    int T_c = ceil(float(src_seq_len) / B_c);

    int group = (query_heads + kv_heads - 1) / kv_heads;

    float softmax_scale = 1.0f / sqrt(float(head_dim));

    int qo_stride_b = target_seq_len * query_heads * head_dim;
    int qo_stride_s = query_heads * head_dim;
    int qo_stride_n = head_dim;

    int kv_stride_b = src_seq_len * kv_heads * head_dim;
    int kv_stride_s = kv_heads * head_dim;
    int kv_stride_n = head_dim;
    
    T* d_q = nullptr;
    cudaMalloc(&d_q, h_q.size() * sizeof(T));
    cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T), cudaMemcpyHostToDevice);
    
    T* d_k = nullptr;
    cudaMalloc(&d_k, h_k.size() * sizeof(T));
    cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T), cudaMemcpyHostToDevice);
    
    T* d_v = nullptr;
    cudaMalloc(&d_v, h_v.size() * sizeof(T));
    cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T), cudaMemcpyHostToDevice);
    
    T* d_o = nullptr;
    cudaMalloc(&d_o, h_o.size() * sizeof(T));
    cudaMemset(d_o, 0, h_o.size() * sizeof(T));

    const int sram_size = (2 * B_c * head_dim * sizeof(T))
                        + (B_r * head_dim * sizeof(T))
                        + (B_c * B_r * sizeof(T));

    dim3 grid(batch_size, query_heads);
    dim3 block(B_r);

    flashAttentionKernel<T><<<grid, block, sram_size>>>(
        d_q, d_k, d_v, d_o,
        target_seq_len, src_seq_len,
        head_dim, group,
        softmax_scale, is_causal,
        B_r, B_c, T_r, T_c,
        qo_stride_b, qo_stride_s, qo_stride_n,
        kv_stride_b, kv_stride_s, kv_stride_n
    );

    cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
