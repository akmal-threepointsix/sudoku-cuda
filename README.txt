Hardware:
- NVIDIA GeForce GTX 1060 6GB (1280 CUDA cores)
- Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz, 4201 Mhz, 4 Core(s), 8 Logical Processor(s)
- RAM 32.0 GB

Results for 95 hard sudokus:
+------+--------+---------+
|      | CPU    | GPU     |
+------+--------+---------+
| Time | 37 sec | 1.2 sec |
+------+--------+---------+

==51984== Profiling application: sudoku.exe inp.txt
==51984== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.91%  949.32ms      1214  781.98us  46.944us  75.884ms  fillSudoku(unsigned char*, unsigned char*, unsigned int*, bool*)
                    0.04%  421.03us      1309     321ns     223ns  1.8880us  [CUDA memcpy DtoH]
                    0.04%  356.61us       285  1.2510us     160ns  2.9120us  [CUDA memset]
                    0.01%  88.003us        95     926ns     864ns  1.2800us  [CUDA memcpy HtoD]
      API calls:   77.05%  1.03125s      1214  849.46us  65.000us  75.961ms  cudaDeviceSynchronize
                   11.79%  157.84ms       380  415.36us  2.6000us  130.48ms  cudaMalloc
                    6.01%  80.464ms      1404  57.310us  22.400us  541.30us  cudaMemcpy
                    1.91%  25.537ms         1  25.537ms  25.537ms  25.537ms  cuDevicePrimaryCtxRelease
                    1.65%  22.040ms      1214  18.155us  7.1000us  213.90us  cudaLaunchKernel
                    1.29%  17.265ms       380  45.433us  1.9000us  334.60us  cudaFree
                    0.30%  4.0016ms       285  14.040us  1.4000us  191.20us  cudaMemset
                    0.00%  33.200us         1  33.200us  33.200us  33.200us  cuModuleUnload
                    0.00%  14.700us       101     145ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  4.6000us         3  1.5330us     200ns  4.0000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.4000us         2     700ns     200ns  1.2000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuModuleGetLoadingMode
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid