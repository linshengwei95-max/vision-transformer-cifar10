"""Unit 02：单头 Scaled Dot-Product Self-Attention 的完整数据流。"""

import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v):
    # 只转置最后两维，Batch 不变；每个 Query 与同一 Batch 的全部 Key 比较。
    scores = q @ k.transpose(-2, -1)  # [2, 4, 32] @ [2, 32, 4] → [2, 4, 4]。
    scaled_scores = scores / math.sqrt(q.shape[-1])  # d_k = 32，shape 不变。
    # 最后一维对应 Key Token；每行归一化为对全部 4 个 Token 的权重。
    weights = torch.softmax(scaled_scores, dim=-1)
    output = weights @ v  # [2, 4, 4] @ [2, 4, 48] → [2, 4, 48]。
    return scores, scaled_scores, weights, output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 4, 96, device="cpu")

    # 三个独立的 Linear 层从同一输入 X 生成 Q、K、V，均只改变最后一维。
    q_projection = nn.Linear(96, 32)
    k_projection = nn.Linear(96, 32)
    v_projection = nn.Linear(96, 48)
    q = q_projection(x)
    k = k_projection(x)
    v = v_projection(x)
    scores, scaled_scores, weights, output = scaled_dot_product_attention(q, k, v)

    for name, tensor in (
        ("Input X", x),
        ("Q", q),
        ("K", k),
        ("V", v),
        ("Attention scores", scores),
        ("Scaled scores", scaled_scores),
        ("Attention weights", weights),
        ("Attention output", output),
    ):
        print(f"{name}: {list(tensor.shape)}")

    assert tuple(x.shape) == (2, 4, 96)
    assert tuple(q.shape) == (2, 4, 32)
    assert tuple(k.shape) == (2, 4, 32)
    assert tuple(v.shape) == (2, 4, 48)
    assert tuple(scores.shape) == (2, 4, 4)
    assert tuple(scaled_scores.shape) == (2, 4, 4)
    assert tuple(weights.shape) == (2, 4, 4)
    assert tuple(output.shape) == (2, 4, 48)

    row_sums = weights.sum(dim=-1)
    assert torch.isfinite(weights).all() and torch.isfinite(output).all()
    assert torch.allclose(row_sums, torch.ones(2, 4), rtol=1e-5, atol=1e-6)
    print("Attention row sums:", row_sums.tolist())
    print("PASS: all shapes and row normalization.")

    # 本次不使用 Mask；所有 Token 对（含对角线上的自身）都有正的注意力权重。
    assert (weights > 0).all()
    print("PASS: every token attends to all 4 tokens, including itself.")

    # 用每个 Batch 的独立二维矩阵运算，对照批量计算，避免误用 Batch 维。
    for batch_index in range(2):
        reference_scores = q[batch_index] @ k[batch_index].T / math.sqrt(32)
        reference_output = torch.softmax(reference_scores, dim=-1) @ v[batch_index]
        torch.testing.assert_close(output[batch_index], reference_output)

    # 分别只修改一个 Batch；其输出应变化，另一个 Batch 的输出应完全不变。
    for changed_batch in range(2):
        changed_x = x.clone()
        changed_x[changed_batch] += 10.0
        changed_q = q_projection(changed_x)
        changed_k = k_projection(changed_x)
        changed_v = v_projection(changed_x)
        _, _, _, changed_output = scaled_dot_product_attention(changed_q, changed_k, changed_v)
        untouched_batch = 1 - changed_batch
        torch.testing.assert_close(
            changed_output[untouched_batch], output[untouched_batch], rtol=0, atol=0
        )
        assert not torch.allclose(changed_output[changed_batch], output[changed_batch])
    print("PASS: independent per-batch references and two-way batch isolation.")
