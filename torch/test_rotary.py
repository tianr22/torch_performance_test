import torch
from utils import test_bw, parse_input
from platforms import DEVICE
import argparse
from model_info import ModelInfo


use_native_rotary = hasattr(torch, "rotary")


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(DEVICE) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        if torch.__version__[0] == "2":
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :], persistent=False
            )
        else:
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)  # .to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q1 = q * cos
    qz = rotate_half(q)
    q2 = qz * sin
    qq = q1 + q2
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def test_rotary(rotary_emb, mixed_x_layer, nh, dim):
    def e():
        new_tensor_shape = mixed_x_layer.size()[:-1] + (nh, 3 * dim)
        x = mixed_x_layer.view(*new_tensor_shape)

        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(x, 3)
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)
        cos, sin = rotary_emb(value_layer, seq_len=new_tensor_shape[0])
        if use_native_rotary:
            query_layer, key_layer = torch.rotary(query_layer, key_layer, sin, cos)
        else:
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer, key_layer, cos, sin, offset=0
            )

    def t(elapsed_time):
        bw = mixed_x_layer.numel() / elapsed_time
        return f"alg bw {bw * 2 * 1e-9:5.1f} GB/s"

    return e, t


def main(ms, bs, sl, dtype=torch.bfloat16):
    info = ModelInfo(ms)
    dim = info.hidden_dim
    heads = info.heads

    dim_per_head = dim // heads
    rotary_emb = RotaryEmbedding(dim_per_head).to(dtype)
    mixed_x_layer = torch.randn(sl, bs, dim * 3, dtype=dtype, device=DEVICE)

    log_string = f"Rotary Dims {dim:5}/{heads:5}"
    tm, bw = test_bw(*test_rotary(rotary_emb, mixed_x_layer, heads, dim_per_head))
    log_string += f" | mean time {tm * 1e3:6.1f} ms, " + bw

    print(log_string)


if __name__ == "__main__":
    args = parse_input()

    main(args.model_size, args.batch_size, args.seq_len)
