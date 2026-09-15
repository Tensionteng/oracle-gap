import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from chronos import BaseChronosPipeline


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # batch all channels in one predict call: [B, L, C] -> [B*C, L]
        B, L, C = x_enc.shape
        flat = x_enc.permute(0, 2, 1).reshape(B * C, L)
        output = self.model.predict(flat, prediction_length=self.pred_len)
        output = output.mean(dim=1)  # [B*C, pred_len]
        dec_out = output.reshape(B, C, self.pred_len).permute(0, 2, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
