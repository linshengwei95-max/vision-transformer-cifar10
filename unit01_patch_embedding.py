"""Unit 01：验证图像 Batch、Patch 序列和 Patch Embedding 的形状。"""

import torch
from torch import nn


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 5
    channels = 3
    image_size = 48
    patch_size = 6
    embedding_dim = 96

    assert image_size % patch_size == 0
    num_patches = (image_size // patch_size) ** 2  # 8×8 = 64 个 Patch。
    patch_dim = channels * patch_size * patch_size  # 3×6×6 = 108。

    images = torch.randn(batch_size, channels, image_size, image_size, device="cpu")
    print("Image batch:", list(images.shape))

    # 高、宽方向窗口和步长均为 6，得到非重叠网格 [5, 3, 8, 8, 6, 6]。
    patch_grid = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    print("Patch grid:", list(patch_grid.shape))

    # 调整为 [B, 网格行, 网格列, C, Patch高, Patch宽]，再展平为 [5, 64, 108]。
    # Patch 按从上到下、从左到右排列，每个 Patch 的特征按 C、H、W 展平。
    patch_sequence = patch_grid.permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size, num_patches, patch_dim
    )
    print("Flattened patch sequence:", list(patch_sequence.shape))

    # 只创建一个 nn.Linear(108, 96)，所有 Batch、所有 Patch 共享该层。
    linear = nn.Linear(patch_dim, embedding_dim)
    embeddings = linear(patch_sequence)  # 只改变最后一维：[5, 64, 108] → [5, 64, 96]。
    print("Patch embeddings:", list(embeddings.shape))

    assert tuple(images.shape) == (5, 3, 48, 48)
    assert tuple(patch_grid.shape) == (5, 3, 8, 8, 6, 6)
    assert tuple(patch_sequence.shape) == (5, 64, 108)
    assert tuple(embeddings.shape) == (5, 64, 96)
    assert images.shape[0] == patch_sequence.shape[0] == embeddings.shape[0] == 5
    assert patch_sequence.shape[1] == embeddings.shape[1] == 64
    print("PASS: all shape assertions passed; one shared Linear(108, 96).")
