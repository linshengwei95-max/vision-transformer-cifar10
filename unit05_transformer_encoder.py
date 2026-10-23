"""可复用的 Pre-Norm Transformer Encoder Block，以及子层正确性检查。"""

import copy
import math

import torch
from torch import nn

from unit03_multi_head_attention import MultiHeadSelfAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=96, num_heads=4, mlp_ratio=4):
        super().__init__()
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (embed_dim, num_heads)):
            raise ValueError("embed_dim 和 num_heads 必须为正整数")
        if (not isinstance(mlp_ratio, (int, float)) or isinstance(mlp_ratio, bool)
                or not math.isfinite(mlp_ratio) or mlp_ratio <= 0):
            raise ValueError("mlp_ratio 必须是有限的正数")
        hidden_dim = int(embed_dim * mlp_ratio)
        if hidden_dim < 1:
            raise ValueError("MLP 隐藏维度必须至少为 1")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x, return_intermediates=False):
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError("Encoder 输入必须为 [Batch, Token, embed_dim]")
        # Pre-Norm：先归一化，再计算子层；残差路径使用未归一化的输入。
        norm1_output = self.norm1(x)
        attention_result = self.attention(norm1_output, return_intermediates=return_intermediates)
        if return_intermediates:
            attention_output, attention_stages = attention_result
        else:
            attention_output = attention_result
        after_attention = x + attention_output
        norm2_output = self.norm2(after_attention)
        mlp_output = self.mlp(norm2_output)
        output = after_attention + mlp_output
        if return_intermediates:
            return output, {
                "Input": x, "Norm1": norm1_output, "Attention": attention_output,
                "Attention stages": attention_stages, "After attention residual": after_attention,
                "Norm2": norm2_output, "MLP": mlp_output, "Output": output,
            }
        return output


def run_encoder_tests():
    """任务 009 测试 B：子层、残差和独立实现的数值对照，不更新主模型参数。"""
    torch.manual_seed(1)
    block = TransformerEncoderBlock().eval()
    x = torch.randn(2, 65, 96)
    mlp_shapes = {}

    def capture_hidden(module, inputs, output):
        mlp_shapes["hidden"] = tuple(output.shape)

    handle = block.mlp[0].register_forward_hook(capture_hidden)
    try:
        with torch.no_grad():
            output, stages = block(x, return_intermediates=True)
    finally:
        handle.remove()
    assert mlp_shapes["hidden"] == (2, 65, 384)
    for name in ("Norm1", "Attention", "After attention residual", "Norm2", "MLP", "Output"):
        assert tuple(stages[name].shape) == (2, 65, 96)
        assert torch.isfinite(stages[name]).all()
    assert tuple(stages["Attention stages"]["Q heads"].shape) == (2, 4, 65, 24)

    # 以每个 Token 自己的特征均值 / 方差为独立参考，检查 LayerNorm 的归一化轴。
    for norm, norm_input, norm_output in (
        (block.norm1, x, stages["Norm1"]),
        (block.norm2, stages["After attention residual"], stages["Norm2"]),
    ):
        assert norm.normalized_shape == (96,)
        mean = norm_input.mean(dim=-1, keepdim=True)
        variance = norm_input.var(dim=-1, unbiased=False, keepdim=True)
        expected = (norm_input - mean) / torch.sqrt(variance + norm.eps)
        expected = expected * norm.weight + norm.bias
        torch.testing.assert_close(norm_output, expected, rtol=1e-5, atol=1e-6)

    # 只在测试副本上将两个子层输出置零；若残差连接正确，输出必须严格等于输入。
    identity_block = copy.deepcopy(block)
    with torch.no_grad():
        identity_block.attention.output_projection.weight.zero_()
        identity_block.attention.output_projection.bias.zero_()
        identity_block.mlp[2].weight.zero_()
        identity_block.mlp[2].bias.zero_()
        torch.testing.assert_close(identity_block(x), x, rtol=0, atol=0)

    # 与 PyTorch 自带的同权重 Pre-Norm Encoder 对照，独立检查 Head / Token 组织。
    reference = nn.TransformerEncoderLayer(
        d_model=96, nhead=4, dim_feedforward=384, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    ).eval()
    attention = block.attention
    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(torch.cat([
            attention.q_projection.weight, attention.k_projection.weight, attention.v_projection.weight
        ]))
        reference.self_attn.in_proj_bias.copy_(torch.cat([
            attention.q_projection.bias, attention.k_projection.bias, attention.v_projection.bias
        ]))
        reference.self_attn.out_proj.load_state_dict(attention.output_projection.state_dict())
        reference.linear1.load_state_dict(block.mlp[0].state_dict())
        reference.linear2.load_state_dict(block.mlp[2].state_dict())
        reference.norm1.load_state_dict(block.norm1.state_dict())
        reference.norm2.load_state_dict(block.norm2.state_dict())
        reference_output = reference(x)
        torch.testing.assert_close(output, reference_output, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(block(x), output, rtol=0, atol=0)
    max_error = (output - reference_output).abs().max().item()
    print(f"PASS B: LayerNorm feature axis; MLP [2,65,96]->[2,65,384]->[2,65,96]; residual identity; PyTorch reference max error={max_error:.9g}.")
    return max_error


if __name__ == "__main__":
    run_encoder_tests()
