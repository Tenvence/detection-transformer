import torch
import torch.nn as nn


class SinePositionEmbedding(nn.Module):
    def __init__(self, num_features):
        super(SinePositionEmbedding, self).__init__()
        self.num_features = num_features // 2
        self.temperature = 10000

    @torch.no_grad()
    def forward(self, pad_mask):
        not_mask = ~pad_mask

        y_embed = not_mask.cumsum(dim=1).float()
        x_embed = not_mask.cumsum(dim=2).float()

        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6)
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6)

        dim_t = torch.arange(self.num_features).float().to(device=pad_mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_features)  # 2 * (dim_t // 2) maps odd number 2i+1 to 2i, and maps even number 2i to 2i

        pos_embed_x = x_embed[..., None] / dim_t  # shape: [batch_size, h, w, dim_t]
        pos_embed_y = y_embed[..., None] / dim_t

        # after "stack", each pair of sin and cos are in the same dimension. after "flatten", sin and cos appear alternately
        pos_embed_x = torch.stack((pos_embed_x[:, :, :, 0::2].sin(), pos_embed_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed_y = torch.stack((pos_embed_y[:, :, :, 0::2].sin(), pos_embed_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        return pos_embed
