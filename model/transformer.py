import copy

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoders=6, num_decoders=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoders)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoders)

        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pad_mask, pos_embed, query_embed):
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pad_mask, pos_embed)
        hs = self.decoder(tgt, memory, pad_mask, pos_embed, query_embed)

        return hs


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, pad_mask, pos):
        for layer in self.layers:
            src = layer(src, pad_mask, pos)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, memory, pad_mask, pos, query_pos):
        for layer in self.layers:
            tgt = layer(tgt, memory, pad_mask, pos, query_pos)
        return tgt


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention_layer = nn.MultiheadAttention(d_model, nhead, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(dropout)

        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, src, pad_mask, pos):
        attention_out = self.self_attention_layer(query=src + pos, key=src + pos, value=src, key_padding_mask=pad_mask, need_weights=False)
        src = src + self.attention_dropout(attention_out)
        src = self.attention_norm(src)

        ffn_out = self.ffn(src)
        src = src + self.ffn_dropout(ffn_out)
        src = self.ffn_norm(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention_layer = nn.MultiheadAttention(d_model, nhead, dropout)
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.encoder_decoder_attention_layer = nn.MultiheadAttention(d_model, nhead, dropout)
        self.encoder_decoder_attention_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attention_dropout = nn.Dropout(dropout)

        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, pad_mask, pos, query_pos):
        query = tgt + query_pos
        key = tgt + query_pos
        value = tgt

        self_attention_out = self.self_attention_layer(query, key, value, need_weights=False)
        tgt = tgt + self.self_attention_dropout(self_attention_out)
        tgt = self.self_attention_norm(tgt)

        query = tgt + query_pos
        key = memory + pos
        value = memory

        encoder_decoder_attention_out = self.encoder_decoder_attention_layer(query, key, value, key_padding_mask=pad_mask, need_weights=False)
        tgt = tgt + self.encoder_decoder_attention_dropout(encoder_decoder_attention_out)
        tgt = self.encoder_decoder_attention_norm(tgt)

        ffn_out = self.ffn(tgt)
        tgt = tgt + self.ffn_dropout(ffn_out)
        tgt = self.ffn_norm(tgt)

        return tgt


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.in_linear = nn.Linear(in_features=d_model, out_features=dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.out_linear = nn.Linear(in_features=dim_feedforward, out_features=d_model)

    def forward(self, x):
        x = self.in_linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_linear(x)
        return x
