import torch.nn as nn
import torch.nn.functional as func

from model.position_embedding import SinePositionEmbedding


class Detr(nn.Module):
    def __init__(self, backbone, transformer, num_channels, num_classes, num_queries):
        super(Detr, self).__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.proj_conv = nn.Conv2d(in_channels=num_channels, out_channels=transformer.d_model, kernel_size=1)

        self.class_pred_head = nn.Linear(in_features=transformer.d_model, out_features=num_classes)
        self.bbox_pred_head = nn.Sequential(
            nn.Linear(in_features=transformer.d_model, out_features=transformer.d_model),
            nn.ReLU(),
            nn.Linear(in_features=transformer.d_model, out_features=transformer.d_model),
            nn.ReLU(),
            nn.Linear(in_features=transformer.d_model, out_features=4),
            nn.Sigmoid()
        )

        self.pos_embed = SinePositionEmbedding(num_features=transformer.d_model)
        self.query_embed = nn.Embedding(num_embeddings=num_queries, embedding_dim=transformer.d_model)

        self.num_classes = num_classes
        self.num_queries = num_queries

    def forward(self, x, pad_mask):
        x = self.backbone(x)
        x = self.proj_conv(x)

        src = x.flatten(start_dim=2).permute(2, 0, 1)  # [B, C, H, W] -> [B, C, HW] -> [HW, B, C]
        pad_mask = func.interpolate(pad_mask[:, None, :, :].float(), size=x.shape[-2:]).bool().squeeze()  # [B, H, W]
        pos_embed = self.pos_embed(pad_mask).flatten(start_dim=2).permute(2, 0, 1)  # [B, C, H, W] -> [B, C, HW] -> [HW, B, C]
        pad_mask = pad_mask.flatten(start_dim=1)  # [B, H, W] -> [B, HW]
        query_embed = self.query_embed.weight.unsqueeze(dim=1).repeat(1, x.shape[0], 1)  # [num_queries, d_model] -> [num_queries, 1, d_model] -> [num_queries, B, d_model]

        hs = self.transformer(src, pad_mask, pos_embed, query_embed).transpose(dim0=0, dim1=1)  # [num_queries, B, d_model] -> [B, num_queries, d_model]

        logist_pred = self.class_pred_head(hs)  # [B, num_queries, d_model] -> [B, num_queries, num_classes]
        bboxes_pred = self.bbox_pred_head(hs)  # [B, num_queries, d_model] -> [B, num_queries, 4]

        return logist_pred, bboxes_pred
