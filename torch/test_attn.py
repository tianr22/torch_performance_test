import torch
from functools import partial
from platforms import DEVICE, PEAK
from utils import test_bw, parse_input
from model_info import ModelInfo


try:
    # FlashAttention (1.x)
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    from flash_attn.flash_attn_triton import flash_attn_func
except ImportError:
    flash_attn_unpadded_func = None
    flash_attn_func = None

try:
    # FlashAttention-2
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    flash_attn_varlen_qkvpacked_func = None


def prepare_data(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_k,
    d,
    attn_func,
    dtype=torch.float16,
):
    assert attn_func is not None, "attn_func is None"
    attn_func_name = attn_func.__name__
    packed = "qkvpacked" in attn_func_name
    varlen = "varlen" in attn_func_name or "unpadded" in attn_func_name
    if packed:
        assert seqlen_q == seqlen_k, "seqlen_q != seqlen_k for packed attention"
        qkv = torch.randn(
            batch_size * seqlen_q,
            3,
            nheads,
            d,
            device=DEVICE,
            dtype=dtype,
            requires_grad=True,
        )
    else:
        q = torch.randn(
            [batch_size * seqlen_q, nheads, d],
            device=DEVICE,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.randn(
            [batch_size * seqlen_k, nheads_k, d],
            device=DEVICE,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.randn(
            [batch_size * seqlen_k, nheads_k, d],
            device=DEVICE,
            dtype=dtype,
            requires_grad=True,
        )
    if varlen:
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=DEVICE,
        )
        if not packed:
            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * seqlen_k,
                step=seqlen_k,
                dtype=torch.int32,
                device=DEVICE,
            )
            input_data = (q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k)
        else:
            input_data = (qkv, cu_seqlens_q, seqlen_q)
    else:
        if packed:
            input_data = qkv
        else:
            input_data = (q, k, v)
    flop = 12 * max(nheads, nheads_k) * d * (max(seqlen_q, seqlen_k) ** 2)
    return input_data, flop


def test_attn(func, input_data, flop):
    def e():
        y = func(*input_data)
        y.sum().backward()

    def t(elapsed_time):
        flops = flop / elapsed_time
        mfu = flops * 1e-12 / PEAK
        return f"{flops * 1e-12:7.3f} TFLOPs, MFU {mfu * 100:4.1f}%"

    return e, t


class AttentionLayer(torch.nn.Module):
    def __init__(self, modelinfo, **kwargs):
        super().__init__()
        self.hidden_dim = modelinfo.hidden_dim
        self.heads = modelinfo.heads
        self.heads_k = modelinfo.heads_k
        self.head_dim = self.hidden_dim // self.heads
        self.q_proj = torch.nn.Linear(
            self.hidden_dim, self.heads * self.head_dim, **kwargs
        )
        self.k_proj = torch.nn.Linear(
            self.hidden_dim, self.heads_k * self.head_dim, **kwargs
        )
        self.v_proj = torch.nn.Linear(
            self.hidden_dim, self.heads_k * self.head_dim, **kwargs
        )
        self.o_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim, **kwargs)

    def forward(self, h):
        # [batch_size * seqlen_q, nheads, d],
        assert len(h.shape) == 3, f"expected 3D tensor, got {len(h.shape)}"
        batch_size, seqlen_q, _ = h.shape
        seqlen_k = seqlen_q
        h = h.view(-1, self.hidden_dim)
        q = self.q_proj(h).view(-1, self.heads, self.head_dim)
        k = self.k_proj(h).view(-1, self.heads_k, self.head_dim)
        v = self.v_proj(h).view(-1, self.heads_k, self.head_dim)
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=DEVICE,
        )
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=DEVICE,
        )
        return self.o_proj(
            flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k
            ).view(-1, self.hidden_dim)
        )


def test_attn_layer(layer, x):
    def e():
        y = layer(x)
        y.sum().backward()

    def t(elapsed_time):
        return f""

    return e, t


def main(
    ms,
    bs,
    seqlen_q,
    seqlen_k=4096,
    nheads=32,
    nheads_k=32,
    func=flash_attn_varlen_func,
):
    assert seqlen_q == seqlen_k
    info = ModelInfo(ms)
    overall_dim = info.hidden_dim
    d = overall_dim // nheads
    input_data, flop = prepare_data(
        bs,
        seqlen_q,
        seqlen_k,
        nheads,
        nheads_k,
        d,
        func,
    )
    t, bw = test_bw(*test_attn(func, input_data, flop))

    log_string = f"AttnOP batchsize {bs:3} seqlen_q {seqlen_q:5} seqlen_k {seqlen_k:5} nheads {nheads:3} nheads_k {nheads_k:3} d {d:5} func {func.__name__}"
    log_string += f" | time {t * 1e3:6.1f} ms, " + bw
    print(log_string)

    # attn layer
    x = torch.randn(bs, seqlen_q, overall_dim, dtype=torch.float16, device=DEVICE)
    attn_layer = AttentionLayer(info, dtype=torch.float16, device=DEVICE)
    t, bw = test_bw(*test_attn_layer(attn_layer, x))

    log_string = f"AttnLayer batchsize {bs:3} seqlen_q {seqlen_q:5} seqlen_k {seqlen_k:5} nheads {nheads:3} nheads_k {nheads_k:3} d {d:5} func {func.__name__}"
    log_string += f" | time {t * 1e3:6.1f} ms, " + bw
    print(log_string)


if __name__ == "__main__":
    args = parse_input()

    main(args.model_size, args.batch_size, args.seq_len)
