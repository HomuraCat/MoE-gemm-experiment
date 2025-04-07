# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import triton
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.activation import SiluAndMul
from v0_moe_fused import fused_moe as fused_moe_grouped
from v2_moe_fused import fused_moe as fused_moe_col
import time

def torch_moe(a, w1, w2, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(B * topk_ids.shape[1],
                      w2.shape[1],
                      dtype=a.dtype,
                      device=a.device)

    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1)).sum(dim=1)


def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.cuda.manual_seed(3227)

    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    score = torch.softmax(score, dim=-1)

    topk_weight, topk_ids = torch.topk(score, topk)

    start = time.time()
    triton_output_gl = fused_moe_grouped(a, w1, w2, topk_weight, topk_ids, False)
    end = time.time()
    gl_time = end - start
    gl_time = gl_time * 1000
    print("Grouped Launch Time (us): ", gl_time)

    start = time.time()
    triton_output_cm = fused_moe_col(a, w1, w2, topk_weight, topk_ids, False)
    end = time.time()
    cm_major_time = end - start
    cm_major_time = cm_major_time * 1000
    print("Columm Major Time (us): ", cm_major_time)

    torch_base = torch_moe(a, w1, w2, topk_weight, topk_ids)
    torch.testing.assert_close(triton_output_cm, torch_base, atol=1e-2, rtol=0)

    # print(f"{triton_output_cm=}\n")
    # print(f"{triton_output_gl=}\n")

    print(f"Col Major Speedup {((gl_time - cm_major_time)/(gl_time))*100}")

# 定义块大小配置
# configs = [
#     {'block_m': m, 'block_n': n, 'block_k': k} for m in [32, 64, 128, 256] for n in [32, 64, 128, 256] for k in [32, 64, 128, 256] if (n*m*k <= 32*128*128) # limited by memory
# ]
# m_range = [2**i for i in range(0, 10)]

#configs = [{'block_m': 64, 'block_n': 64, 'block_k': 32}]
configs = [{'block_m': 32, 'block_n': 32, 'block_k': 256}]
m_range = [2]

# 生成 line_vals 和 line_names
line_vals = [f"{kernel}_{i}" for i in range(len(configs)) for kernel in ['cm']]
line_names = [f"{kernel.upper()} block_m={configs[i]['block_m']} block_n={configs[i]['block_n']} block_k={configs[i]['block_k']}" for i in range(len(configs)) for kernel in ['cm']]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['m'],  # x 轴为 m
        x_vals=m_range,  # m 的不同值
        line_arg='provider',  # 用于区分不同曲线的参数
        line_vals=line_vals,  # 内核类型和 config 索引的组合
        line_names=line_names,  # 每条曲线的可读名称
        ylabel="TFLOPS",  # y 轴标签
        plot_name="fused-moe-blocksize-performance",  # 图表名称
        args={},
    )
)
def benchmark(m, provider):
    # 解析 provider 以获取内核类型和 config 索引
    kernel_type, config_idx = provider.split('_')
    config_idx = int(config_idx)
    config = configs[config_idx]

    # 固定参数
    n = 14336 // 2
    k = 4096
    e = 8
    topk = 2
    dtype = torch.float16

    # 设置随机种子以确保可重复性
    torch.cuda.manual_seed(3227)

    # 生成输入张量
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10
    score = torch.randn((m, e), device='cuda', dtype=dtype)
    score = torch.softmax(score, dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)

    # 根据内核类型运行基准测试
    quantiles = [0.5, 0.2, 0.8]
    if kernel_type == 'cm':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_moe_col(a, w1, w2, topk_weight, topk_ids, config, False),
            quantiles=quantiles
        )
    elif kernel_type == 'gl':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_moe_grouped(a, w1, w2, topk_weight, topk_ids, False),
            quantiles=quantiles
        )
    else:
        raise ValueError(f"未知的内核类型: {kernel_type}")

    # 计算 TFLOPS
    perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True, save_path='./')