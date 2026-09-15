import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from models.DescriptorHead import DescriptorHead, ChannelDescriptorHead, ValueMtpHead

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST (https://arxiv.org/pdf/2211.14730.pdf) with an optional auxiliary
    head on the pooled encoder hidden state. Minimal-diff clone of
    models/PatchTST.py:
      - forecast() additionally returns hidden = mean over (variates, patch
        tokens) of the encoder output -> [B, d_model];
      - forward(..., return_hidden=True) returns (forecast, aux) where aux is
        the head output (configs.aux_head == 'desc' or 'valuemtp');
      - with aux_head == 'none' (or without return_hidden) the module behaves
        exactly like the stock PatchTST.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

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
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

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
                # concat z with the per-channel pooled tokens, project back
                # (the projection is learned from scratch)
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
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # pooled hidden for the auxiliary head: mean over variates and patch
        # tokens; in channel mode keep the variate axis (mean over patch
        # tokens per channel) -> [bs x nvars x d_model]
        if self.desc_mode == 'channel':
            hidden = enc_out.mean(dim=2)
        else:
            hidden = enc_out.mean(dim=(1, 2))  # [bs x d_model]

        aux = None
        if self.aux_head is not None and self.desc_cond != 'none':
            # cascade conditioning ("predict the space before the values"):
            # the descriptor representation z modulates the patch tokens
            # before the stock FlattenHead; no stop-gradient, so the task loss
            # also flows through the descriptor trunk
            aux, z = self.aux_head(hidden, return_z=True)
            enc_out = self._condition_tokens(enc_out, z)

        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, (aux if aux is not None else hidden)

    def _condition_tokens(self, enc_out, z):
        """enc_out: [bs, C, patch_num, d_model]; z: [bs, z_dim] (pooled) or
        [bs, C, z_dim] (channel mode)."""
        B, C, P, D = enc_out.shape
        if self.desc_cond == 'concat':
            x_c = enc_out.mean(dim=2)                     # [B, C, d_model]
            if z.dim() == 2:
                z = z.unsqueeze(1).expand(-1, C, -1)      # broadcast over channels
            fused = self.cond_proj(torch.cat([x_c, z], dim=-1))
            return enc_out + fused.unsqueeze(2)           # broadcast over patches
        elif self.desc_cond == 'film':
            scale = self.cond_scale(z)
            shift = self.cond_shift(z)
            if z.dim() == 2:
                scale, shift = scale.unsqueeze(1).unsqueeze(1), shift.unsqueeze(1).unsqueeze(1)
            else:
                scale, shift = scale.unsqueeze(2), shift.unsqueeze(2)
            return enc_out * (1 + scale) + shift
        return enc_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_hidden=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, hidden = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            dec_out = dec_out[:, -self.pred_len:, :]  # [B, L, D]
            if return_hidden:
                # in desc_cond mode forecast() already ran the head (the main
                # path depends on z), so hidden is already the aux dict
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
