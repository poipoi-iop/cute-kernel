# FA2 vs flash_attn

GPU: NVIDIA GeForce RTX 3090; FP16, non-causal, dropout=0, MHA.
PyTorch 2.8.0+cu128; flash_attn 2.8.3.post1; CUDA 12.8.
Timing: graph, warmup=10, iterations=50, samples=9.
Both providers allocate O through PyTorch; graph timing excludes allocation and Python submission.
TFLOP/s counts QK and PV only (4*B*H*Sq*Sk*D), excludes softmax.
Full-output FP32 reference with TF32 disabled; atol=0.002, rtol=0.002.

| B | Sq | Sk | H | D | FA2 us | flash_attn us | FA2 / library latency |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 64 | 8 | 32 | 2.785 | 6.410 | 0.435x |
| 1 | 64 | 64 | 8 | 64 | 3.666 | 10.199 | 0.359x |
| 1 | 128 | 128 | 8 | 64 | 5.448 | 9.011 | 0.605x |
| 1 | 512 | 512 | 8 | 64 | 16.404 | 23.245 | 0.706x |
| 1 | 1024 | 1024 | 8 | 64 | 50.708 | 53.166 | 0.954x |
| 1 | 2048 | 2048 | 8 | 64 | 180.634 | 164.557 | 1.098x |
| 1 | 4096 | 4096 | 8 | 64 | 570.798 | 635.064 | 0.899x |
| 4 | 1024 | 1024 | 8 | 64 | 153.846 | 168.161 | 0.915x |
| 1 | 64 | 64 | 8 | 128 | 4.832 | 8.110 | 0.596x |
| 1 | 128 | 128 | 8 | 128 | 7.475 | 13.046 | 0.573x |
| 1 | 512 | 512 | 8 | 128 | 23.982 | 35.205 | 0.681x |
| 1 | 1024 | 1024 | 8 | 128 | 83.333 | 85.340 | 0.976x |
| 1 | 2048 | 2048 | 8 | 128 | 325.179 | 307.425 | 1.058x |
| 1 | 4096 | 4096 | 8 | 128 | 1132.605 | 1198.141 | 0.945x |
| 4 | 1024 | 1024 | 8 | 128 | 297.103 | 310.292 | 0.957x |
| 1 | 128 | 512 | 8 | 64 | 14.152 | 10.977 | 1.289x |
| 1 | 1024 | 256 | 8 | 128 | 23.491 | 25.272 | 0.929x |
