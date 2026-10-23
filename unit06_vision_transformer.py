"""小型、可配置的 Vision Transformer 分类模型及任务 009 的 A～E 验证。"""

import torch
from torch import nn

from unit03_multi_head_attention import merge_heads
from unit04_position_embedding import ViTInputEmbedding
from unit05_transformer_encoder import TransformerEncoderBlock, run_encoder_tests


class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=96,
                 num_heads=4, depth=2, mlp_ratio=4, num_classes=10):
        super().__init__()
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (image_size, patch_size, in_channels, embed_dim, num_heads, depth, num_classes)):
            raise ValueError("尺寸、通道、维度、Head 数、深度和类别数必须为正整数")
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        # 直接复用 Unit 04；仅将本模型的 in_channels 映射到已有 channels 接口。
        self.input_embedding = ViTInputEmbedding(
            image_size=image_size, patch_size=patch_size, channels=in_channels, embed_dim=embed_dim
        )
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images, return_intermediates=False):
        embedded = self.input_embedding(images, return_intermediates=return_intermediates)
        if return_intermediates:
            x, input_stages = embedded
            block_stages = []
        else:
            x = embedded
        for block in self.blocks:
            if return_intermediates:
                x, stages = block(x, return_intermediates=True)
                block_stages.append(stages)
            else:
                x = block(x)
        normalized = self.norm(x)
        cls_features = normalized[:, 0, :]  # 取最终 CLS 表示，保留 Batch 维度。
        logits = self.head(cls_features)  # 返回原始 Logits，交叉熵内部处理归一化。
        if return_intermediates:
            return logits, {
                "Input embedding": input_stages, "Encoder blocks": block_stages,
                "Final norm": normalized, "CLS": cls_features, "Logits": logits,
            }
        return logits


def run_demo_and_tests():
    torch.manual_seed(0)
    model = VisionTransformer().eval()
    images = torch.randn(2, 3, 32, 32)

    # A：验证默认 CIFAR-10 尺寸，以及旧实验的 48/6 初始化配置。
    with torch.no_grad():
        for current_model, side, patch_size in (
            (model, 32, 4), (VisionTransformer(image_size=48, patch_size=6).eval(), 48, 6)
        ):
            current_images = images if side == 32 else torch.randn(2, 3, side, side)
            logits, stages = current_model(current_images, return_intermediates=True)
            inputs = stages["Input embedding"]
            for name, shape in {
                "Patch tokens": (2, 64, 96), "Tokens with CLS": (2, 65, 96),
                "Final sequence": (2, 65, 96),
            }.items():
                assert tuple(inputs[name].shape) == shape and torch.isfinite(inputs[name]).all()
                print(f"A {side}/{patch_size} {name}: {list(inputs[name].shape)}")
            for index, block in enumerate(stages["Encoder blocks"]):
                for name, tensor in block.items():
                    if isinstance(tensor, torch.Tensor):
                        assert tuple(tensor.shape) == (2, 65, 96) and torch.isfinite(tensor).all()
                for tensor in block["Attention stages"].values():
                    assert torch.isfinite(tensor).all()
                print(f"A {side}/{patch_size} Encoder {index}: {list(block['Input'].shape)} -> {list(block['Output'].shape)}")
            assert tuple(stages["Final norm"].shape) == (2, 65, 96)
            assert tuple(stages["CLS"].shape) == (2, 96)
            assert tuple(logits.shape) == (2, 10)
            assert torch.isfinite(stages["Final norm"]).all() and torch.isfinite(logits).all()
            print(f"A {side}/{patch_size} Final norm / CLS / Logits: {list(stages['Final norm'].shape)} / {list(stages['CLS'].shape)} / {list(logits.shape)}")
            torch.testing.assert_close(current_model(current_images), logits, rtol=0, atol=0)
    print("PASS A: both 32/4 and 48/6 configurations; traced and normal forward agree.")

    reference_error = run_encoder_tests()  # 复用 Unit 05 的子层测试，不另建测试框架。

    # C：一次随机分类损失反向传播；检查全部参数的注册、梯度和主模型参数未更新。
    model.train()
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    labels = torch.tensor([1, 7], dtype=torch.long)
    loss = nn.functional.cross_entropy(model(images), labels)
    assert torch.isfinite(loss)
    loss.backward()
    registered = dict(model.named_parameters())
    required = ["input_embedding.patch_projection", "input_embedding.cls_token",
                "input_embedding.position_embedding", "head"]
    for index in range(model.depth):
        required.extend(f"blocks.{index}.attention.{projection}" for projection in (
            "q_projection", "k_projection", "v_projection", "output_projection"
        ))
        required.extend((f"blocks.{index}.mlp.0", f"blocks.{index}.mlp.2"))
    for prefix in required:
        assert any(name == prefix or name.startswith(prefix + ".") for name in registered)
    all_zero_gradients = []
    key_bias_maxima = []
    for name, parameter in registered.items():
        assert parameter.requires_grad and parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.equal(parameter.detach(), snapshots[name]), name
        if torch.count_nonzero(parameter.grad).item() == 0:
            all_zero_gradients.append(name)
        if name.endswith("attention.k_projection.bias"):
            key_bias_maxima.append(parameter.grad.abs().max().item())
    print(f"PASS C: loss={loss.item():.9g}; all {len(registered)} parameter tensors have finite gradients; no parameter updates.")
    print("C exactly all-zero gradient tensors:", all_zero_gradients)
    print("C per-block key-bias max |gradient|:", key_bias_maxima)
    # Key 的公共 bias 给每个 Query 的所有 Key 分数加同一常数，Softmax 理论上不变。
    # 所以这类 bias 的梯度可为零或浮点近零，不能据此认定断图，也不能只靠非零梯度证明正确。

    # D：eval 重复性、不同输入、单独 / 批量推理与双向 Batch 隔离。
    model.eval()
    with torch.no_grad():
        baseline = model(images)
        torch.testing.assert_close(model(images), baseline, rtol=0, atol=0)
        assert torch.isfinite(baseline).all()
        assert not torch.allclose(baseline[0], baseline[1])
        for batch_index in range(2):
            torch.testing.assert_close(model(images[batch_index:batch_index + 1])[0], baseline[batch_index], rtol=1e-4, atol=1e-5)
        for changed_batch in range(2):
            changed_images = images.clone()
            changed_images[changed_batch] += torch.randn_like(changed_images[changed_batch])
            changed_logits = model(changed_images)
            untouched = 1 - changed_batch
            assert torch.isfinite(changed_logits).all()
            torch.testing.assert_close(changed_logits[untouched], baseline[untouched], rtol=0, atol=0)
            assert not torch.allclose(changed_logits[changed_batch], baseline[changed_batch])
        # 复用 Unit 03 的正式合并函数与明确数值反例，不重跑整个旧实验。
        tiny = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
        assert torch.equal(merge_heads(tiny), torch.tensor([[[1, 2, 5, 6], [3, 4, 7, 8]]]))
        assert not torch.equal(merge_heads(tiny), tiny.reshape(1, 2, 4))
    print("PASS D: finite/distinct logits; exact eval repeatability and two-way batch isolation; per-image agreement; correct head merge order.")

    # E：实际结构、注册关系、配置可调性，以及不合法配置的拒绝行为。
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    component_counts = {
        "input_embedding": sum(p.numel() for p in model.input_embedding.parameters()),
        "encoder_blocks": sum(p.numel() for p in model.blocks.parameters()),
        "final_norm": sum(p.numel() for p in model.norm.parameters()),
        "classification_head": sum(p.numel() for p in model.head.parameters()),
    }
    assert total_parameters == sum(component_counts.values())
    assert total_parameters == 235882 and len(model.blocks) == 2
    assert len({id(block) for block in model.blocks}) == 2
    assert all(block.attention.num_heads == 4 and block.attention.head_dim == 24
               and block.hidden_dim == 384 for block in model.blocks)
    assert model.head.in_features == 96 and model.head.out_features == 10
    with torch.no_grad():
        alternative = VisionTransformer(image_size=16, patch_size=4, in_channels=1,
                                        embed_dim=48, num_heads=3, depth=1, mlp_ratio=2, num_classes=7).eval()
        assert tuple(alternative(torch.randn(3, 1, 16, 16)).shape) == (3, 7)
    for invalid_config in ({"embed_dim": 95}, {"image_size": 31}, {"depth": 0},
                           {"num_classes": 0}, {"mlp_ratio": 0}):
        try:
            VisionTransformer(**invalid_config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"不合法配置未被拒绝：{invalid_config}")
    print(f"PASS E: parameters={total_parameters}; blocks=2; heads=4; head_dim=24; classes=10; configurable valid/invalid cases checked.")
    print("E component parameter counts:", component_counts)
    print(f"PASS: A-E completed; loss={loss.item():.9g}; encoder reference error={reference_error:.9g}; no optimizer, dataset download, or training loop.")


if __name__ == "__main__":
    run_demo_and_tests()
