"""Unit 04：Patch Embedding、可学习 CLS / 位置嵌入与 A～F 验证。

Unit 01 的实现位于 __main__ 内，没有可导入的 Patch Embedding 接口。
本文件仅沿用其 unfold / permute / reshape / Linear 逻辑，不修改旧脚本。
"""

import torch
from torch import nn


class ViTInputEmbedding(nn.Module):
    def __init__(self, image_size=48, patch_size=6, channels=3, embed_dim=96):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("图像尺寸和 Patch 尺寸必须为正，且图像尺寸能被 Patch 尺寸整除")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_projection = nn.Linear(channels * patch_size * patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.empty(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, images, return_intermediates=False):
        if images.ndim != 4 or tuple(images.shape[1:]) != (
            self.channels, self.image_size, self.image_size
        ):
            raise ValueError("输入必须为 [Batch, channels, image_size, image_size]")
        batch_size = images.shape[0]

        # 与 Unit 01 一致：非重叠网格 → 行优先的 Patch 序列 → 共享 Linear。
        patch_grid = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        flat_patches = patch_grid.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size, self.num_patches, self.channels * self.patch_size ** 2
        )
        patch_tokens = self.patch_projection(flat_patches)  # [2, 64, 96]。

        # expand 共享同一个 CLS 参数；沿 Token 维度拼接到序列首位。
        expanded_cls = self.cls_token.expand(batch_size, -1, -1)  # [2, 1, 96]。
        tokens_with_cls = torch.cat((expanded_cls, patch_tokens), dim=1)  # [2, 65, 96]。
        # [1, 65, 96] 沿 Batch 广播，索引 0 的位置向量也加到 CLS 上。
        final_sequence = tokens_with_cls + self.position_embedding

        if return_intermediates:
            return final_sequence, {
                "Images": images,
                "Patch grid": patch_grid,
                "Flat patches": flat_patches,
                "Patch tokens": patch_tokens,
                "CLS parameter": self.cls_token,
                "Expanded CLS": expanded_cls,
                "Tokens with CLS": tokens_with_cls,
                "Position parameter": self.position_embedding,
                "Final sequence": final_sequence,
            }
        return final_sequence


def run_demo_and_tests():
    torch.manual_seed(0)
    model = ViTInputEmbedding()
    images = torch.randn(2, 3, 48, 48, device="cpu")
    final_sequence, stages = model(images, return_intermediates=True)

    # 测试 A：完整数据流的实际 shape 和有限值。
    expected_shapes = {
        "Images": (2, 3, 48, 48),
        "Patch grid": (2, 3, 8, 8, 6, 6),
        "Flat patches": (2, 64, 108),
        "Patch tokens": (2, 64, 96),
        "CLS parameter": (1, 1, 96),
        "Expanded CLS": (2, 1, 96),
        "Tokens with CLS": (2, 65, 96),
        "Position parameter": (1, 65, 96),
        "Final sequence": (2, 65, 96),
    }
    for name, expected_shape in expected_shapes.items():
        assert tuple(stages[name].shape) == expected_shape
        assert torch.isfinite(stages[name]).all()
        print(f"{name}: {list(stages[name].shape)}")
    print("PASS A: all shapes and finite values.")

    # 测试 B：CLS 在索引 0，原 Patch 序列按原顺序保留在索引 1～64。
    torch.testing.assert_close(stages["Tokens with CLS"][:, :1], stages["Expanded CLS"], rtol=0, atol=0)
    torch.testing.assert_close(stages["Tokens with CLS"][:, 1:], stages["Patch tokens"], rtol=0, atol=0)
    # 直接图像切片独立对照，验证两张图的全部 Patch 内容及行优先顺序。
    reference_flat = torch.stack([
        images[:, :, row:row + 6, col:col + 6].reshape(2, 108)
        for row in range(0, 48, 6)
        for col in range(0, 48, 6)
    ], dim=1)
    torch.testing.assert_close(stages["Flat patches"], reference_flat, rtol=0, atol=0)
    torch.testing.assert_close(stages["Patch tokens"], model.patch_projection(reference_flat))
    print("PASS B: CLS is first; all 128 patches retain content and row-major order.")

    # 测试 C：每个 Batch 的每个位置都加上同一张位置表的对应向量。
    added_positions = final_sequence - stages["Tokens with CLS"]
    expected_positions = model.position_embedding.expand(2, -1, -1)
    torch.testing.assert_close(added_positions, expected_positions, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(added_positions[0], added_positions[1], rtol=1e-5, atol=1e-6)
    for batch_index in range(2):
        expected_batch = stages["Tokens with CLS"][batch_index] + model.position_embedding[0]
        torch.testing.assert_close(final_sequence[batch_index], expected_batch, rtol=0, atol=0)
    max_position_error = (added_positions - expected_positions).abs().max().item()
    print(f"PASS C: position vectors broadcast across batches; max subtraction error = {max_position_error:.9g}.")

    # 测试 D：明确整数值演示位置与内容的对应关系，不仅比较 shape。
    same_content = torch.tensor([[[10, 20], [10, 20]]])
    small_positions = torch.tensor([[[1, 2], [3, 4]]])
    different_positions = same_content + small_positions
    assert torch.equal(different_positions, torch.tensor([[[11, 22], [13, 24]]]))
    assert not torch.equal(different_positions[0, 0], different_positions[0, 1])
    print("D same content + different positions:", different_positions.tolist())

    small_content = torch.tensor([[[10, 20], [30, 40]]])
    order = torch.tensor([1, 0])
    shuffle_then_add = small_content[:, order] + small_positions
    add_then_shuffle = (small_content + small_positions)[:, order]
    assert torch.equal(shuffle_then_add, torch.tensor([[[31, 42], [13, 24]]]))
    assert torch.equal(add_then_shuffle, torch.tensor([[[33, 44], [11, 22]]]))
    assert shuffle_then_add.shape == add_then_shuffle.shape == (1, 2, 2)
    assert not torch.equal(shuffle_then_add, add_then_shuffle)
    print("D content:", small_content.tolist(), "positions:", small_positions.tolist(), "order:", order.tolist())
    print("D shuffle content, then add fixed positions:", shuffle_then_add.tolist())
    print("D add positions, then shuffle whole tokens:", add_then_shuffle.tolist())
    print("PASS D: position assignment and permutation give distinct numeric results.")

    # 测试 E：确认参数注册与梯度；仅 backward 一次，不创建优化器或更新参数。
    registered = dict(model.named_parameters())
    snapshots = {name: parameter.detach().clone() for name, parameter in registered.items()}
    for name in ("cls_token", "position_embedding"):
        parameter = getattr(model, name)
        assert isinstance(parameter, nn.Parameter) and parameter.requires_grad
        assert registered[name] is parameter
    final_sequence.sum().backward()
    for name in ("cls_token", "position_embedding"):
        parameter = registered[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        # Batch=2，求和探针在两个 Batch 中对共享参数各贡献 1，故梯度为 2。
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2.0), rtol=0, atol=0)
        print(f"E {name}: registered=True, requires_grad=True, grad shape={list(parameter.grad.shape)}, grad values={parameter.grad.unique().tolist()}.")
    for name, parameter in registered.items():
        assert torch.equal(parameter.detach(), snapshots[name])
    print("PASS E: one backward check; every model parameter remains unchanged.")

    # 测试 F：独立参数副本构造另一种设计：仅 Patch 加位置，再拼接无额外位置的 CLS。
    # 副本沿用经典分支的相同初值，使两个分支的 Patch 部分可以直接数值对照。
    alternative_cls = nn.Parameter(model.cls_token.detach().clone())
    patch_only_positions = nn.Parameter(model.position_embedding[:, 1:].detach().clone())
    positioned_patches = stages["Patch tokens"].detach() + patch_only_positions
    alternative_sequence = torch.cat((alternative_cls.expand(2, -1, -1), positioned_patches), dim=1)
    assert tuple(patch_only_positions.shape) == (1, 64, 96)
    assert tuple(positioned_patches.shape) == (2, 64, 96)
    assert tuple(alternative_sequence.shape) == (2, 65, 96)
    assert torch.isfinite(alternative_sequence).all()
    torch.testing.assert_close(alternative_sequence[:, :1], alternative_cls.expand(2, -1, -1), rtol=0, atol=0)
    torch.testing.assert_close(alternative_sequence[:, 1:], final_sequence[:, 1:].detach(), rtol=0, atol=0)
    cls_difference = final_sequence[:, :1].detach() - alternative_sequence[:, :1]
    torch.testing.assert_close(cls_difference, model.position_embedding[:, :1].expand(2, -1, -1), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(alternative_sequence[:, :1], final_sequence[:, :1].detach())
    print("F patch-only position parameter:", list(patch_only_positions.shape))
    print("F alternative sequence:", list(alternative_sequence.shape))
    print("PASS F: same positioned patches; alternative CLS has no position vector.")
    print("PASS: all tests A-F completed; no optimizer or parameter updates.")


if __name__ == "__main__":
    run_demo_and_tests()
