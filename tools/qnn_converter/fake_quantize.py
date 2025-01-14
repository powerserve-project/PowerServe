import torch
from torch import Tensor


class Quantizer:
    CLIP_MIN = 1e-5
    CLIP_MAX = 1e6

    def __init__(
        self,
        bitwidth: int,
        symmetric: bool = True,
        per_channel: bool = False,
        block_size: int = -1,
        requires_calibration: bool = False,
    ):
        self.bitwidth = bitwidth
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.block_size = block_size
        self.requires_calibration = requires_calibration

        self.min_val: Tensor | None = None
        self.max_val: Tensor | None = None
        self.quants: Tensor | None = None
        self.scales: Tensor | None = None

    def get_min_max(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.per_channel:
            if self.block_size == -1:
                y = x
            else:
                y = x.reshape(-1, self.block_size)
            min_val = torch.amin(y, dim=-1, keepdim=True)
            max_val = torch.amax(y, dim=-1, keepdim=True)
        else:
            y = x.view(-1)
            min_val = torch.amin(y, dim=-1)
            max_val = torch.amax(y, dim=-1)

        return min_val, max_val

    def update_min_max(self, x: Tensor):
        if not self.requires_calibration:
            return

        min_val, max_val = self.get_min_max(x)

        if self.min_val is None:
            self.min_val = min_val
        else:
            torch.minimum(self.min_val, min_val, out=self.min_val)

        if self.max_val is None:
            self.max_val = max_val
        else:
            torch.maximum(self.max_val, max_val, out=self.max_val)

    def quantize_with_min_max(self, x: Tensor, min_val: Tensor, max_val: Tensor, save_qaunts: bool = False) -> Tensor:
        assert min_val.shape == max_val.shape

        if self.per_channel:
            if self.block_size == -1:
                y = x
            else:
                y = x.reshape(-1, self.block_size)

            assert y.shape[:-1] == min_val.shape[:-1]
        else:
            y = x.view(-1)

        y = y.clip(min=min_val, max=max_val).to(torch.float32)
        min_val = min_val.to(torch.float32)
        max_val = max_val.to(torch.float32)

        if self.symmetric:
            alpha = torch.maximum(torch.abs(min_val), torch.abs(max_val))
            q_min = -(2 ** (self.bitwidth - 1))
            q_max = 2 ** (self.bitwidth - 1) - 1

            scales = (alpha / q_max).abs().clip(min=self.CLIP_MIN, max=self.CLIP_MAX)
            inverted_scales = 1 / scales
            quants = (y * inverted_scales).round().clip(q_min, q_max)
            y = quants * scales
            # print(y.dtype, alpha, q_min, scale)

            if save_qaunts:
                self.scales = scales.cpu()
                self.quants = quants.cpu()
        else:
            alpha = max_val - min_val
            beta = min_val
            q_min = 0
            q_max = 2**self.bitwidth - 1
            scale = (alpha / q_max).abs().clip(min=self.CLIP_MIN, max=self.CLIP_MAX)
            inverted_scale = 1 / scale

            offset = (beta * inverted_scale).round()
            y = (((y - beta) * inverted_scale).round().clip(q_min, q_max) + offset) * scale
            # print(scale.item(), offset.item())

            # NOTE: This cannot exactly represent zero, but accuracy is much better and llama.cpp uses this
            # y = (((y - beta) * inverted_scale).round()) * scale + beta
            # print(scale.item(), beta.item())

            # print(y.dtype, max_val, min_val, alpha, beta, q_max, scale)

        return y.type_as(x).reshape_as(x)

    def quantize(self, x: Tensor, save_qaunts: bool = False) -> Tensor:
        if self.min_val is None:
            assert self.max_val is None
            assert not self.requires_calibration
            min_val, max_val = self.get_min_max(x)
        else:
            assert self.max_val is not None
            assert self.requires_calibration
            min_val = self.min_val
            max_val = self.max_val

        return self.quantize_with_min_max(x, min_val, max_val, save_qaunts=save_qaunts)

    def __call__(self, x: Tensor) -> Tensor:
        return self.quantize(x)
