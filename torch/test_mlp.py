import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils import test_bw, parse_input
from platforms import DEVICE, PEAK
from model_info import ModelInfo


use_native_silu = hasattr(torch, "fused_silu")


class Dense1to4(nn.Module):
    def __init__(self, dim, ffn_dim, **kwargs):
        super().__init__()
        self.dense = nn.Linear(dim, ffn_dim * 2, bias=False, **kwargs)
        self.numel = dim * ffn_dim * 2

    def forward(self, x):
        return self.dense(x)


class Dense4to1(nn.Module):
    def __init__(self, dim, ffn_dim, **kwargs):
        super().__init__()
        self.dense = nn.Linear(ffn_dim, dim, bias=False, **kwargs)
        self.numel = dim * ffn_dim

    def forward(self, x):
        return self.dense(x)


class MyMLP(nn.Module):
    def __init__(self, dim, ffn_dim, **kwargs):
        super().__init__()
        self.dense_hto4h = Dense1to4(dim, ffn_dim, **kwargs)
        self.dense_4htoh = Dense4to1(dim, ffn_dim, **kwargs)
        self.numel = dim * ffn_dim * 3

    def forward(self, x):
        x = self.dense_hto4h(x)
        x = torch.chunk(x, 2, dim=-1)
        if use_native_silu:
            x = torch.fused_silu(x[0].contiguous(), x[1].contiguous())
        else:
            x = F.silu(x[0]) * x[1]
        x = self.dense_4htoh(x)
        return x


def test_mlp(layer, x):
    def e():
        y = layer(x)
        y.sum().backward()

    def t(elapsed_time):
        flops = 2 * 3 * layer.numel * x.shape[0] * x.shape[1] / elapsed_time
        mfu = flops * 1e-12 / PEAK
        return f"{flops * 1e-12:7.3f} TFLOPs, MFU {mfu * 100:4.1f}%"

    return e, t


def main(ms, bs, sl, dtype=torch.bfloat16):
    info = ModelInfo(ms)
    dim = info.hidden_dim
    ffn_dim = info.ffn
    x = torch.randn(bs, sl, dim, dtype=dtype, device=DEVICE)
    x.requires_grad = True
    for mp in (1, 2, 4, 8):
        log_string = f"FFN Dims {dim:5}/{ffn_dim:5} mp {mp}"

        layer = MyMLP(dim, ffn_dim // mp, dtype=dtype, device=DEVICE)
        tm_all, bw = test_bw(*test_mlp(layer, x))
        log_string += f" | mean time {tm_all * 1e3:6.1f} ms, " + bw

        layer = Dense1to4(dim, ffn_dim // mp, dtype=dtype, device=DEVICE)
        tm_d1, bw = test_bw(*test_mlp(layer, x))
        log_string += f" | hto4h mean time {tm_d1 * 1e3:6.1f} ms, " + bw

        layer = Dense4to1(dim, ffn_dim // mp, dtype=dtype, device=DEVICE)
        y = torch.randn(bs, sl, ffn_dim // mp, dtype=dtype, device=DEVICE)
        y.requires_grad = True
        tm_d2, bw = test_bw(*test_mlp(layer, y))
        log_string += f" | 4htoh mean time {tm_d2 * 1e3:6.1f} ms, " + bw

        tm_rest = tm_all - tm_d1 - tm_d2
        rest_frac = tm_rest / tm_all
        log_string += (
            f" | silu mean time {tm_rest * 1e3:6.1f} ms {rest_frac * 100:4.1f}%"
        )

        print(log_string)


if __name__ == "__main__":
    args = parse_input()

    main(args.model_size, args.batch_size, args.seq_len)
