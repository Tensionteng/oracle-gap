import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from models.DescriptorHead import DescriptorHead, ValueMtpHead


class Model(nn.Module):
    """
    DLinear (https://arxiv.org/pdf/2205.13504.pdf) with an optional auxiliary
    head, serving as the negative-control backbone (no deep hidden state).
    Minimal-diff clone of models/DLinear.py:
      - hidden = concat of the time-pooled (mean over seq_len) seasonal and
        trend decomposition inputs -> [B, 2C], passed through a small
        projection (Linear(2C, 64) + GELU);
      - forward(..., return_hidden=True) returns (forecast, aux) where aux is
        the head output (configs.aux_head == 'desc' or 'valuemtp');
      - with aux_head == 'none' (or without return_hidden) the module behaves
        exactly like the stock DLinear.
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

        # Auxiliary head on a small projection of the pooled decomposition
        # inputs (descriptor / value-MTP). hidden_proj is always built so that
        # the return_hidden path (e.g. analysis/probe_descriptor.py) also works
        # with aux_head == 'none'; load base DLinear checkpoints with
        # strict=False to skip it.
        self.aux_head_type = getattr(configs, 'aux_head', 'none')
        if getattr(configs, 'desc_mode', 'pooled') == 'channel' and self.aux_head_type != 'none':
            raise ValueError('DLinearDesc does not support desc_mode=channel '
                             '(its hidden is a channel-pooled projection by design)')
        if getattr(configs, 'desc_cond', 'none') != 'none' and self.aux_head_type != 'none':
            raise ValueError('DLinearDesc does not support desc_cond '
                             '(negative-control backbone has no token path to condition)')
        self.d_hidden = 64
        self.hidden_proj = nn.Sequential(
            nn.Linear(2 * self.channels, self.d_hidden), nn.GELU())
        if self.aux_head_type == 'desc':
            self.aux_head = DescriptorHead(self.d_hidden, dropout=configs.dropout)
        elif self.aux_head_type == 'valuemtp':
            self.aux_head = ValueMtpHead(self.d_hidden, self.pred_len * self.channels,
                                         dropout=configs.dropout)
        else:
            self.aux_head = None

    def encoder(self, x, return_hidden=False):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if return_hidden:
            # pooled decomposition inputs: [B, C] seasonal mean + [B, C] trend mean
            pooled = torch.cat([seasonal_init.mean(dim=2), trend_init.mean(dim=2)], dim=1)
            hidden = self.hidden_proj(pooled)  # [B, d_hidden]
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        if return_hidden:
            return x.permute(0, 2, 1), hidden
        return x.permute(0, 2, 1)

    def forecast(self, x_enc, return_hidden=False):
        # Encoder
        return self.encoder(x_enc, return_hidden=return_hidden)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_hidden=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if return_hidden:
                dec_out, hidden = self.forecast(x_enc, return_hidden=True)
                dec_out = dec_out[:, -self.pred_len:, :]  # [B, L, D]
                aux = self.aux_head(hidden) if self.aux_head is not None else hidden
                return dec_out, aux
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
