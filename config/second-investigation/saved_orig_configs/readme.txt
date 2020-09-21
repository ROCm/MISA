

 1. Removed gemm_k_per_block == 32,  gemm_k_per_block == 64 configurations

 2. Added configurations where tensor_b_thread_lengths[ c1e ]  = 4 or tensor_b_thread_lengths[ c1e ] = 2 (if 4 is not possible)


