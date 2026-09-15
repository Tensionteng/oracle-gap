import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


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


class GatedFullAttention(nn.Module):
    """Full softmax attention via F.scaled_dot_product_attention with two optional
    selectivity mechanisms (mechanism-ablation study, logs/mechanism):

    - variant='none'    : plain softmax attention. Numerically equal to
                          layers.SelfAttention_Family.FullAttention(mask_flag=False);
                          the SDPA kernel is used for memory/speed only.
    - variant='denbias' : "softmax off-by-one": an extra learnable null logit b_h
                          (per head) is appended to each row of attention logits;
                          its softmax mass is discarded (the null key has a zero
                          value vector), giving attention a trainable
                          "attend-to-nothing" exit.
    - variant='recmask' : hard recency mask: each query may only attend to the
                          most recent `recmask_keys` key positions (patches).

    output_attention=True switches to an explicit einsum softmax path (eval only,
    small batches) so attention maps can be inspected; for 'denbias' the returned
    map covers real keys only (rows may sum to < 1).
    """

    def __init__(self, n_heads, attention_dropout=0.1, variant='none',
                 recmask_keys=None, output_attention=False):
        super(GatedFullAttention, self).__init__()
        assert variant in ('none', 'denbias', 'recmask')
        self.n_heads = n_heads
        self.variant = variant
        self.recmask_keys = recmask_keys
        self.output_attention = output_attention
        self.dropout_p = attention_dropout
        if variant == 'denbias':
            self.null_logit = nn.Parameter(torch.zeros(n_heads))

    def _keep(self, S):
        keep = self.recmask_keys if self.recmask_keys is not None else S
        return min(keep, S)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        q = queries.transpose(1, 2)  # [B, H, L, E]
        k = keys.transpose(1, 2)     # [B, H, S, E]
        v = values.transpose(1, 2)   # [B, H, S, D]

        if self.output_attention:
            # explicit path for attention-map inspection (eval only)
            scale = 1. / sqrt(E)
            scores = torch.einsum('bhle,bhse->bhls', q, k) * scale
            if self.variant == 'denbias':
                nb = self.null_logit.view(1, H, 1, 1).expand(B, H, L, 1)
                scores = torch.cat([scores, nb], dim=-1)
            elif self.variant == 'recmask':
                keep = self._keep(S)
                if keep < S:
                    scores[..., :-keep] = float('-inf')
            A = torch.softmax(scores, dim=-1)
            if self.variant == 'denbias':
                A = A[..., :-1]  # discard null mass (it contributes a zero vector)
            out = torch.einsum('bhls,bhsd->bhld', A, v)
            return out.transpose(1, 2).contiguous(), A

        bias = None
        if self.variant == 'denbias':
            k = torch.cat([k, k.new_zeros(B, H, 1, E)], dim=2)
            v = torch.cat([v, v.new_zeros(B, H, 1, D)], dim=2)
            # additive bias: null-key logit = b_h (zero key -> q.k = 0)
            bias = q.new_zeros(1, H, 1, S + 1)
            bias[0, :, 0, -1] = self.null_logit
        elif self.variant == 'recmask':
            keep = self._keep(S)
            if keep < S:
                bias = q.new_zeros(1, 1, 1, S)
                bias[..., :-keep] = float('-inf')

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout_p if self.training else 0.0)
        return out.transpose(1, 2).contiguous(), None


class Model(nn.Module):
    """PatchTST with selectable attention gating (mechanism ablation).
    Architecture, normalization and head are identical to models/PatchTST.py;
    only the inner attention module differs (GatedFullAttention).
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        variant = getattr(configs, 'attn_variant', 'none')
        recmask_window = getattr(configs, 'recmask_window', 336)
        recmask_keys = -(-recmask_window // stride)  # ceil(window/stride) patches
        output_attention = getattr(configs, 'output_attention', False)

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        GatedFullAttention(configs.n_heads,
                                           attention_dropout=configs.dropout,
                                           variant=variant,
                                           recmask_keys=recmask_keys,
                                           output_attention=output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
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
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
