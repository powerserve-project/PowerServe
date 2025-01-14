#!/usr/bin/python
import argparse
import sys
from pathlib import Path

import gguf
import safetensors
import torch
from torch import Tensor, nn


qtype_map: dict[str, gguf.GGMLQuantizationType] = {
    "f32": gguf.GGMLQuantizationType.F32,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}


class Loader:
    """Helper class to load weight tensors from safetensors"""

    def __init__(self, folder: Path):
        """Load tensors from safetensor files and create a mapping between tensor names and tensors"""

        self.tensor_map: dict[str, Tensor] = {}
        for model_shard_file in folder.glob("*.safetensors"):
            tensors = safetensors.safe_open(model_shard_file, "pt")
            for name in tensors.keys():
                self.tensor_map[name] = tensors.get_tensor(name)

    def __contains__(self, name: str) -> bool:
        return name in self.tensor_map

    def load(self, dest: nn.Module | Tensor, name: str, transposed: bool = False):
        """Look up tensor in tensor map and copy data to destination tensor"""

        tensor = self.tensor_map[name]

        target = None
        if isinstance(dest, nn.Module):
            target = dest.weight.data
        elif isinstance(dest, Tensor):
            target = dest.data
        else:
            raise RuntimeError

        if transposed:
            tensor = tensor.T

        assert target.shape == tensor.shape, f"Expect {tuple(target.shape)}, got {tuple(tensor.shape)}"
        target.copy_(tensor.to(torch.float32))


class Exporter:
    def __init__(self, embd: Tensor, src_path: Path, out_path: Path, out_type=gguf.GGMLQuantizationType.F32) -> None:
        self.embd = embd
        self.src_path = src_path
        self.out_path = out_path
        self.out_type = out_type
        self.new_name = "token_embd.weight"
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="qwen2_vl")

    def write_embd(self):
        self.prepare_tensors()
        self.prepare_metadata()
        self.gguf_writer.write_header_to_file(path=self.out_path)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    def prepare_tensors(self):
        if self.embd.dtype not in (torch.float16, torch.float32):
            self.embd = self.embd.to(torch.float32)

        data = self.embd.squeeze().numpy()
        if len(data.shape) == 0:
            data = self.embd.numpy()

        data = gguf.quants.quantize(data, self.out_type)
        self.gguf_writer.add_tensor(self.new_name, data, raw_dtype=self.out_type)

    def prepare_metadata(self):
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, self.src_path, None, total_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--out-type", type=str, choices=["f32", "q8_0"], default="f32")
    args = parser.parse_args()

    loader = Loader(args.model_path)
    embd_name = "model.embed_tokens.weight" if "model.embed_tokens.weight" in loader else "tok_embeddings.weight"
    embd_tensor = loader.tensor_map[embd_name]

    exporter = Exporter(embd_tensor, args.model_path, args.out_path, qtype_map[args.out_type])
    exporter.write_embd()


main()
