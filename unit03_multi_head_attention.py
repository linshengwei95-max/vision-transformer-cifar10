"""Unit 03：显式实现 Multi-Head Self-Attention，并执行 A～E 验证。"""

import math

import torch
from torch import nn

from unit02_scaled_dot_product_attention import scaled_dot_product_attention


def merge_heads(head_output):
    """[B, H, N, D] → [B, N, H, D] → [B, N, H*D]。"""
    batch_size, num_heads, num_tokens, head_dim = head_output.shape
    # 先将同一 Token 的各个 Head 放到一起，再拼接其特征。
    token_heads = head_output.permute(0, 2, 1, 3)
    return token_heads.reshape(batch_size, num_tokens, num_heads * head_dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=96, num_heads=4):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim 和 num_heads 必须为正，且 embed_dim 能被 num_heads 整除")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 本次为 96 / 4 = 24。
        self.q_projection = nn.Linear(embed_dim, embed_dim)
        self.k_projection = nn.Linear(embed_dim, embed_dim)
        self.v_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_intermediates=False):
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError("输入必须为 [Batch, Token, embed_dim]")
        batch_size, num_tokens, _ = x.shape
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        # 先拆最后一维：[2, 6, 96] → [2, 6, 4, 24]。
        split_shape = (batch_size, num_tokens, self.num_heads, self.head_dim)
        q_split, k_split, v_split = q.reshape(split_shape), k.reshape(split_shape), v.reshape(split_shape)
        # 再将 Head 放到 Token 前面：[2, 4, 6, 24]。
        q_heads = q_split.permute(0, 2, 1, 3)
        k_heads = k_split.permute(0, 2, 1, 3)
        v_heads = v_split.permute(0, 2, 1, 3)

        # 复用 Unit 02：Q @ K.transpose(-2, -1)，除以 sqrt(24)，Softmax(dim=-1)，再乘 V。
        # 前两维 [Batch, Head] 保持独立；Score 为 [2, 4, 6, 6]。
        scores, scaled_scores, weights, head_output = scaled_dot_product_attention(
            q_heads, k_heads, v_heads
        )
        merged = merge_heads(head_output)  # [2, 4, 6, 24] → [2, 6, 96]。
        output = self.output_projection(merged)  # 每个 Token 的 96 维拼接特征再做投影。

        if return_intermediates:
            return output, {
                "X": x, "Q": q, "K": k, "V": v,
                "Q split": q_split, "K split": k_split, "V split": v_split,
                "Q heads": q_heads, "K heads": k_heads, "V heads": v_heads,
                "Scores": scores, "Scaled scores": scaled_scores, "Weights": weights,
                "Head output": head_output,
                "Merge permute": head_output.permute(0, 2, 1, 3),
                "Merged heads": merged, "Output": output,
            }
        return output


def run_demo_and_tests():
    torch.manual_seed(0)
    model = MultiHeadSelfAttention(embed_dim=96, num_heads=4)
    x = torch.randn(2, 6, 96, device="cpu")

    # 仅做前向演示和验证，不计算梯度或进行训练。
    with torch.no_grad():
        output, stages = model(x, return_intermediates=True)

        # 测试 A：检查所有关键阶段的实际 shape 和有限值。
        expected_shapes = {
            "X": (2, 6, 96), "Q": (2, 6, 96), "K": (2, 6, 96), "V": (2, 6, 96),
            "Q split": (2, 6, 4, 24), "K split": (2, 6, 4, 24), "V split": (2, 6, 4, 24),
            "Q heads": (2, 4, 6, 24), "K heads": (2, 4, 6, 24), "V heads": (2, 4, 6, 24),
            "Scores": (2, 4, 6, 6), "Scaled scores": (2, 4, 6, 6), "Weights": (2, 4, 6, 6),
            "Head output": (2, 4, 6, 24), "Merge permute": (2, 6, 4, 24),
            "Merged heads": (2, 6, 96), "Output": (2, 6, 96),
        }
        for name, expected_shape in expected_shapes.items():
            assert tuple(stages[name].shape) == expected_shape
            assert torch.isfinite(stages[name]).all()
            print(f"{name}: {list(stages[name].shape)}")

        # 独立二维计算对照每个 Batch、每个 Head，验证 Head 之间不混合注意力。
        for batch_index in range(2):
            for head_index in range(4):
                q_head = stages["Q heads"][batch_index, head_index]
                k_head = stages["K heads"][batch_index, head_index]
                v_head = stages["V heads"][batch_index, head_index]
                reference_weights = torch.softmax(q_head @ k_head.T / math.sqrt(24), dim=-1)
                reference_output = reference_weights @ v_head
                torch.testing.assert_close(stages["Weights"][batch_index, head_index], reference_weights)
                torch.testing.assert_close(stages["Head output"][batch_index, head_index], reference_output)
        print("PASS A: all shapes, finite values, and independent per-head references.")

        # 测试 B：每张图、每个 Head、每个 Query 的权重行和都约为 1。
        row_sums = stages["Weights"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6)
        max_row_error = (row_sums - 1).abs().max().item()
        print(f"PASS B: 48 attention rows normalized; max error = {max_row_error:.9g}.")

        # 测试 C：两个方向分别改变一个 Batch，另一个 Batch 的输出必须完全不变。
        for changed_batch in range(2):
            changed_x = x.clone()
            changed_x[changed_batch] += 10.0
            changed_output = model(changed_x)
            untouched_batch = 1 - changed_batch
            torch.testing.assert_close(
                changed_output[untouched_batch], output[untouched_batch], rtol=0, atol=0
            )
            assert not torch.allclose(changed_output[changed_batch], output[changed_batch])
            unchanged_error = (changed_output[untouched_batch] - output[untouched_batch]).abs().max().item()
            print(f"PASS C: change batch {changed_batch}; other batch max difference = {unchanged_error}.")

        # 测试 D：明确数值验证同一 Token 的跨 Head 拼接；两种方法最终 shape 相同。
        tiny = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])  # [B=1, H=2, N=2, D=2]。
        correct = merge_heads(tiny)  # 使用正式模块实际调用的合并函数。
        wrong = tiny.reshape(1, 2, 4)
        expected = torch.tensor([[[1, 2, 5, 6], [3, 4, 7, 8]]])
        assert tuple(tiny.shape) == (1, 2, 2, 2)
        assert tuple(correct.shape) == tuple(wrong.shape) == (1, 2, 4)
        assert torch.equal(correct, expected)
        assert not torch.equal(wrong, expected)
        print("D input [Batch, Head, Token, Feature]:", tiny.tolist())
        print("D correct permute + reshape:", correct.tolist())
        print("D wrong direct reshape:", wrong.tolist())
        print("PASS D: identical shapes do not guarantee correct feature order.")

        # 测试 E：检查四个独立投影层、合法配置与输出投影的真实数值计算。
        projections = (model.q_projection, model.k_projection, model.v_projection, model.output_projection)
        assert len({id(layer) for layer in projections}) == 4
        assert all(layer.in_features == layer.out_features == 96 for layer in projections)
        assert model.num_heads == 4 and model.head_dim == 24 and model.embed_dim == 96
        expected_output = stages["Merged heads"] @ model.output_projection.weight.T + model.output_projection.bias
        torch.testing.assert_close(output, expected_output)
        assert tuple(output.shape) == (2, 6, 96)
        for embed_dim, num_heads in ((96, 5), (96, 0), (0, 4)):
            try:
                MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
            except ValueError:
                print(f"E rejected incompatible config: embed_dim={embed_dim}, heads={num_heads}.")
            else:
                raise AssertionError("不兼容的 Head 配置没有被拒绝")
        print(f"PASS E: four Linear(96, 96), head_dim=24, output projection, parameters={sum(p.numel() for p in model.parameters())}.")

    print("PASS: all tests A-E completed.")


if __name__ == "__main__":
    run_demo_and_tests()
