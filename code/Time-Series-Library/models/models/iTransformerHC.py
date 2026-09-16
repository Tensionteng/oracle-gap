import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

"""
iTransformer with Hyper-Connections (arXiv:2409.19606).

hc_mode:
  res - standard residual connection (identical to iTransformer)
  hc  - unconstrained hyper-connections
  mhc - manifold-constrained HC on the Birkhoff polytope (Sinkhorn-Knopp, arXiv:2512.24880)
  ohc - HC constrained on the orthogonal group O(n) via the Cayley transform

At initialization all hc variants degenerate to a pre-norm iTransformer:
alpha = e_1, A = I, beta = 1, readout = e_1.
"""


def sinkhorn(logits, iters=20):
    """Project exp(logits) onto the Birkhoff polytope (doubly stochastic)."""
    m = torch.exp(logits - logits.max())
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True)
        m = m / m.sum(dim=-2, keepdim=True)
    return m


def cayley(p):
    """Cayley parameterization of O(n): S = P - P^T skew-symmetric, Q = (I+S)^{-1}(I-S)."""
    s = p - p.transpose(-1, -2)
    eye = torch.eye(p.size(-1), device=p.device, dtype=p.dtype)
    return torch.linalg.solve(eye + s, eye - s)


class HCConnection(nn.Module):
    """One hyper-connected sublayer: x = alpha^T H (pre-norm) -> y = F(x) -> H <- A H + beta * y."""

    def __init__(self, d_model, n, mode, dropout=0.1, out_dropout=False):
        super(HCConnection, self).__init__()
        self.n = n
        self.mode = mode
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if out_dropout else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros(n))          # aggregation, e_1 at init
        self.alpha.data[0] = 1.0
        self.beta = nn.Parameter(torch.ones(n))            # output injection
        if mode == 'mhc':
            self.theta = nn.Parameter(20.0 * torch.eye(n))  # SK(exp(.)) = I at init
        else:
            init = torch.eye(n) if mode == 'hc' else torch.zeros(n, n)
            self.theta = nn.Parameter(init)                 # A = I at init for both

    def mixing_matrix(self):
        if self.mode == 'mhc':
            return sinkhorn(self.theta)
        if self.mode == 'ohc':
            return cayley(self.theta)
        return self.theta

    def forward(self, H, fn):
        # H: [B, L, n, D]
        x = torch.einsum('blnd,n->bld', H, self.alpha)
        y = self.dropout(fn(self.norm(x)))
        A = self.mixing_matrix()
        return torch.einsum('ij,bljd->blid', A, H) + torch.einsum('i,bld->blid', self.beta, y)


class HCEncoderLayer(nn.Module):
    """Mirror of Transformer_EncDec.EncoderLayer with both residuals replaced by hyper-connections."""

    def __init__(self, attention, d_model, n, mode, d_ff=None, dropout=0.1, activation="relu"):
        super(HCEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.hc_attn = HCConnection(d_model, n, mode, dropout, out_dropout=True)
        self.hc_ffn = HCConnection(d_model, n, mode, dropout, out_dropout=False)

    def _ffn(self, x):
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return y

    def forward(self, H, attn_mask=None):
        H = self.hc_attn(H, lambda x: self.attention(x, x, x, attn_mask=attn_mask)[0])
        H = self.hc_ffn(H, self._ffn)
        return H


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625 (iTransformer)
    Hyper-connection variants: arXiv:2409.19606 (HC), arXiv:2512.24880 (mHC), oHC (orthogonal)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hc_mode = getattr(configs, 'hc_mode', 'res')
        self.hc_expand = getattr(configs, 'hc_expand', 4)
        self.use_hc = self.hc_mode != 'res'
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        if self.use_hc:
            self.encoder_layers = nn.ModuleList(
                [
                    HCEncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False), configs.d_model, configs.n_heads),
                        configs.d_model,
                        self.hc_expand,
                        self.hc_mode,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ]
            )
            self.norm = nn.LayerNorm(configs.d_model)
            self.readout = nn.Parameter(torch.zeros(self.hc_expand))  # e_1 at init
            self.readout.data[0] = 1.0
        else:
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
                    ) for _ in range(configs.e_layers)
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

    def _encode(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, D]
        if not self.use_hc:
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            return enc_out
        H = enc_out.unsqueeze(2).expand(-1, -1, self.hc_expand, -1).contiguous()  # [B, L, n, D]
        for layer in self.encoder_layers:
            H = layer(H, attn_mask=None)
        out = torch.einsum('blnd,n->bld', H, self.readout)
        return self.norm(out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self._encode(x_enc, x_mark_enc)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        enc_out = self._encode(x_enc, x_mark_enc)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        enc_out = self._encode(x_enc, None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self._encode(x_enc, x_mark_enc)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
