import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from models.DescriptorHead import DescriptorHead, ChannelDescriptorHead, ValueMtpHead
import numpy as np


class Model(nn.Module):
    """
    iTransformer (https://arxiv.org/abs/2310.06625) with an optional auxiliary
    head on the pooled variate tokens. Minimal-diff clone of
    models/iTransformer.py:
      - forecast() additionally returns hidden = mean over the C variate
        tokens of the encoder output -> [B, d_model];
      - forward(..., return_hidden=True) returns (forecast, aux) where aux is
        the head output (configs.aux_head == 'desc' or 'valuemtp');
      - with aux_head == 'none' (or without return_hidden) the module behaves
        exactly like the stock iTransformer.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

        # Auxiliary head on the pooled (or per-channel) hidden state
        # (descriptor / value-MTP)
        self.aux_head_type = getattr(configs, 'aux_head', 'none')
        self.desc_mode = getattr(configs, 'desc_mode', 'pooled')
        self.desc_cond = getattr(configs, 'desc_cond', 'none')
        if self.aux_head_type == 'desc':
            if self.desc_mode == 'channel':
                self.aux_head = ChannelDescriptorHead(configs.d_model, dropout=configs.dropout)
            else:
                self.aux_head = DescriptorHead(configs.d_model, dropout=configs.dropout)
        elif self.aux_head_type == 'valuemtp':
            if self.desc_mode == 'channel':
                raise ValueError('desc_mode=channel is only supported for aux_head=desc')
            self.aux_head = ValueMtpHead(configs.d_model, configs.pred_len * configs.enc_in,
                                         dropout=configs.dropout)
        else:
            self.aux_head = None
        # cascade conditioning: descriptor representation z -> main path
        self.z_dim = 128  # DescriptorHead trunk output dim
        if self.desc_cond != 'none':
            if self.aux_head_type != 'desc':
                raise ValueError('desc_cond requires aux_head=desc')
            if self.desc_cond == 'concat':
                # concat z with each variate token, project back (learned
                # from scratch), then the stock projection head
                self.cond_proj = nn.Linear(configs.d_model + self.z_dim, configs.d_model)
            elif self.desc_cond == 'film':
                self.cond_scale = nn.Linear(self.z_dim, configs.d_model)
                self.cond_shift = nn.Linear(self.z_dim, configs.d_model)
            else:
                raise ValueError('unknown desc_cond: {}'.format(self.desc_cond))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # pooled hidden for the auxiliary head: mean over the N variate tokens
        # (trailing tokens, if any, come from time-mark embeddings); in channel
        # mode keep the variate axis -> [B, N, d_model]
        if self.desc_mode == 'channel':
            hidden = enc_out[:, :N, :]
        else:
            hidden = enc_out[:, :N, :].mean(dim=1)  # [B, d_model]

        aux = None
        if self.aux_head is not None and self.desc_cond != 'none':
            # cascade conditioning ("predict the space before the values"):
            # the descriptor representation z modulates the variate tokens
            # before the stock projection; no stop-gradient, so the task loss
            # also flows through the descriptor trunk
            aux, z = self.aux_head(hidden, return_z=True)
            enc_out = self._condition_tokens(enc_out, z, N)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, (aux if aux is not None else hidden)

    def _condition_tokens(self, enc_out, z, N):
        """enc_out: [B, N(+marks), d_model]; z: [B, z_dim] (pooled) or
        [B, N, z_dim] (channel mode). Conditions the N variate tokens only."""
        tok = enc_out[:, :N, :]
        if self.desc_cond == 'concat':
            if z.dim() == 2:
                z = z.unsqueeze(1).expand(-1, N, -1)      # broadcast over tokens
            tok = self.cond_proj(torch.cat([tok, z], dim=-1))
        elif self.desc_cond == 'film':
            scale = self.cond_scale(z)
            shift = self.cond_shift(z)
            if z.dim() == 2:
                scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
            tok = tok * (1 + scale) + shift
        return torch.cat([tok, enc_out[:, N:, :]], dim=1)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_hidden=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, hidden = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            dec_out = dec_out[:, -self.pred_len:, :]  # [B, L, D]
            if return_hidden:
                # in desc_cond mode forecast() already ran the head, so hidden
                # is already the aux dict
                if isinstance(hidden, dict):
                    return dec_out, hidden
                aux = self.aux_head(hidden) if self.aux_head is not None else hidden
                return dec_out, aux
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
