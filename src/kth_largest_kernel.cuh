template <typename T>
__global__ void markKernel(
    const T* d_input, int* d_mark, int* d_mark_equal, size_t size, T pivot
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // >= pivot: 1, < pivot: 0
        d_mark[idx] = (d_input[idx] >= pivot) ? 1 : 0;
        d_mark_equal[idx] = (d_input[idx] == pivot) ? 1 : 0;
    }
}


// Blelloch scan
__global__ void blockScanKernel(
    const int* __restrict__ d_mark,
    int* __restrict__ d_scan,
    int* __restrict__ d_blockSums,
    size_t size
){
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;

    extern __shared__ int temp[];
    temp[tid] = (gid < size) ? d_mark[gid] : 0;
    __syncthreads();

    // 包含式扫描
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int t = 0;
        if (tid >= offset) t = temp[tid - offset];
        __syncthreads();
        temp[tid] += t;
        __syncthreads();
    }

    if (gid < size) {
        d_scan[gid] = temp[tid];
    }
    // 记录每个 block 的和（最后一个线程写）
    if (tid == blockDim.x - 1 && d_blockSums) {
        d_blockSums[blockIdx.x] = temp[tid];
    }
}


__global__ void addOffsetsKernel(
    int* __restrict__ d_scan,
    const int* __restrict__ d_blockOffsets,
    size_t size
){
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;

    if (blockIdx.x > 0 && gid < size) {
        d_scan[gid] += d_blockOffsets[blockIdx.x - 1];
    }
}


// 根据 isLeft 将 >= 或 < pivot 的元素移到前面
template <typename T>
__global__ void scatterAllKernel(
    const T* __restrict__ d_in,
    T* __restrict__ d_out,
    const int* __restrict__ d_mark,
    const int* __restrict__ d_scan,
    size_t size,
    bool isLeft
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (isLeft) {
        // 将 >= pivot 的元素移到前面
        if (d_mark[idx] == 1) {
            int pos = d_scan[idx] - 1;
            d_out[pos] = d_in[idx];
        }
    } else {
        // 将 < pivot 的元素移到前面
        if (d_mark[idx] == 0) {
            int pos = idx - d_scan[idx];
            d_out[pos] = d_in[idx];
        }
    }
}


// 计算前缀和
void prefixSum(int* d_mark, int* d_scan, size_t size) {
    const int BLOCK_SIZE = 256;
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* d_blockSums   = nullptr;
    int* d_blockOffsets = nullptr;

    if (numBlocks > 1) {
        cudaMalloc(&d_blockSums,   numBlocks * sizeof(int));
        cudaMalloc(&d_blockOffsets,numBlocks * sizeof(int));
    }

    blockScanKernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_mark, d_scan, d_blockSums, size
    );
    cudaDeviceSynchronize();

    if (numBlocks > 1) {
        // 对 block sums 再做一次扫描，得到每块的偏移
        prefixSum(d_blockSums, d_blockOffsets, numBlocks);

        // 把偏移加回到每个元素的扫描值中
        addOffsetsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_scan, d_blockOffsets, size
        );
        cudaDeviceSynchronize();

        cudaFree(d_blockSums);
        cudaFree(d_blockOffsets);
    }
}