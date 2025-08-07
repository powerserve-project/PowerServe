import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import onnx
import torch
from graph_params import *
from model_params import *
from onnx import shape_inference
from torch import nn
from torch.nn.utils import skip_init
from tqdm import tqdm
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--model-folder", type=Path, required=True)
parser.add_argument("--model-name", choices=model_map.keys(), required=True)
parser.add_argument("--graph-name", choices=graph_map.keys(), required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--system-prompt-file", type=Path)
parser.add_argument("--prompt-file", type=Path, required=True)
parser.add_argument("--n-model-chunks", type=int, default=4)
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--max-n-tokens", type=int, required=True)
parser.add_argument("--output-folder", type=Path)
parser.add_argument("--fp16-lm-head", action="store_true")
args = parser.parse_args()

torch.manual_seed(42)
torch.set_num_threads(args.n_threads)
device = torch.device(args.device)


def export_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


from llama_model import *


class Sample(NamedTuple):
    inputs: Tuple[torch.Tensor]
    outputs: Tuple[torch.Tensor]


class KVCache:
    def init_kv_caches(self, start_layer_id: int, end_layer_id: int, n_kv_heads: int, head_dim: int, cache_size: int):
        self.start_layer_id = start_layer_id
        self.end_layer_id = end_layer_id
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache_size = cache_size

        self.kv_cache_position = 0
        self.saved_kv: Optional[List[torch.Tensor]] = None

        self.key_t_cache_data = torch.zeros(
            size=(self.n_kv_layers, n_kv_heads, head_dim, cache_size), dtype=torch.float32, device=device
        )
        self.value_cache_data = torch.zeros(
            size=(self.n_kv_layers, n_kv_heads, cache_size, head_dim), dtype=torch.float32, device=device
        )

    @property
    def n_kv_layers(self) -> int:
        return self.end_layer_id - self.start_layer_id

    @property
    def n_kv_caches(self) -> int:
        return self.n_kv_layers * self.n_kv_heads

    @property
    def kv_cache_names(self) -> List[str]:
        return [
            *[
                f"layer_{i}_key_t_cache_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
            *[
                f"layer_{i}_value_cache_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
        ]

    @property
    def kv_names(self) -> List[str]:
        return [
            *[
                f"layer_{i}_key_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
            *[
                f"layer_{i}_value_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
        ]

    @property
    def key_t_caches(self) -> Tuple[torch.Tensor]:
        return tuple(self.key_t_cache_data[i, j] for i in range(self.n_kv_layers) for j in range(self.n_kv_heads))

    @property
    def value_caches(self) -> Tuple[torch.Tensor]:
        return tuple(self.value_cache_data[i, j] for i in range(self.n_kv_layers) for j in range(self.n_kv_heads))

    @property
    def kv_caches(self) -> Tuple[torch.Tensor]:
        return self.key_t_caches + self.value_caches

    def update_kv_caches(self, kv: Tuple[torch.Tensor]):
        assert len(kv) == 2 * self.n_kv_caches
        assert all(kv[0].shape == tensor.shape for tensor in kv)

        batch_size, _ = kv[0].shape
        beg = self.kv_cache_position
        end = self.kv_cache_position + batch_size
        self.kv_cache_position += batch_size
        assert end <= self.cache_size

        keys = kv[: self.n_kv_caches]
        values = kv[self.n_kv_caches :]
        for cache, key in zip(self.key_t_caches, keys):
            cache[:, beg:end] = key.transpose(0, 1)
        for cache, value in zip(self.value_caches, values):
            cache[beg:end, :] = value

    def reset_kv_cache_position(self, position: int):
        self.kv_cache_position = position
        for cache in self.key_t_caches:
            cache[:, position:] = 0
        for cache in self.value_caches:
            cache[position:, :] = 0


class ExportableModule(nn.Module):
    def __init__(self, module_type: str, start_layer_id: int, end_layer_id: int):
        super().__init__()
        self.module_type = module_type
        self.start_layer_id = start_layer_id
        self.end_layer_id = end_layer_id
        self.saved_samples: List[Sample] = []

    def get_inputs(
        self, last_outputs: Tuple[torch.Tensor], attn_bias: torch.Tensor, rope_embeds: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        raise RuntimeError


class LlamaModelChunk(ExportableModule, KVCache):
    """A model chunk consists of consecutive transformer layers"""

    def __init__(
        self,
        start_layer_id: int,
        end_layer_id: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        context_size: int,
        ffn_hidden_dim: int,
        rms_norm_eps: float,
        has_qkv_bias: bool,
        use_drelu: bool,
        cache_size: int,
        n_fp16_heads: int | dict[int, int],
        n_fp16_neurons: int | dict[int, int],
        stat_folder: Path,
    ):
        super().__init__("transformers", start_layer_id, end_layer_id)
        self.n_kv_heads = n_kv_heads

        self.layers = nn.ModuleList()
        for layer_id in range(start_layer_id, end_layer_id):
            if isinstance(n_fp16_heads, int):
                cur_n_fp16_heads = n_fp16_heads
            else:
                cur_n_fp16_heads = n_fp16_heads[layer_id]

            if isinstance(n_fp16_neurons, int):
                cur_n_fp16_neurons = n_fp16_neurons
            else:
                cur_n_fp16_neurons = n_fp16_neurons[layer_id]

            if stat_folder.exists():
                assert stat_folder.is_dir()
                with open(stat_folder / f"attn_{layer_id}_stat.json", "r") as f:
                    attn_head_ids = json.load(f)
                    attn_head_ids = attn_head_ids[:cur_n_fp16_heads]
                with open(stat_folder / f"ffn_{layer_id}_stat.json", "r") as f:
                    ffn_neuron_ids = json.load(f)
                    ffn_neuron_ids = ffn_neuron_ids[:cur_n_fp16_neurons]
            else:
                if 0 < cur_n_fp16_heads < n_heads or 0 < cur_n_fp16_neurons < ffn_hidden_dim:
                    print(f'WARN: "{stat_folder}" does not exist.')
                attn_head_ids = list(range(cur_n_fp16_heads))
                ffn_neuron_ids = list(range(cur_n_fp16_neurons))

            self.layers.append(
                LlamaTransformer(
                    layer_id=layer_id,
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    context_size=context_size,
                    ffn_hidden_dim=ffn_hidden_dim,
                    rms_norm_eps=rms_norm_eps,
                    has_qkv_bias=has_qkv_bias,
                    use_drelu=use_drelu,
                    fp16_head_ids=attn_head_ids,
                    fp16_neuron_ids=ffn_neuron_ids,
                    device=device,
                )
            )

        self.init_kv_caches(
            start_layer_id=start_layer_id,
            end_layer_id=end_layer_id,
            n_kv_heads=n_kv_heads,
            head_dim=(embed_dim // n_heads),
            cache_size=cache_size,
        )

    @property
    def n_layers(self) -> int:
        return self.end_layer_id - self.start_layer_id

    @property
    def layer_ids(self) -> range:
        return range(self.start_layer_id, self.end_layer_id)

    @property
    def input_names(self) -> List[str]:
        return ["x", "attn_bias", "rope_embed_cos", "rope_embed_sin", *self.kv_cache_names]

    @property
    def output_names(self) -> List[str]:
        return ["out", *self.kv_names]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["x", "out", "rope_embed_cos", "rope_embed_sin"]

    def load_weights(self, loader: ModelLoader):
        for layer in self.layers:
            layer.load_weights(loader)

    def get_inputs(
        self, last_outputs: Tuple[torch.Tensor], attn_bias: torch.Tensor, rope_embeds: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        return (last_outputs[0], attn_bias, rope_embeds[0], rope_embeds[1], *self.kv_caches)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        rope_embed_cos: torch.Tensor,
        rope_embed_sin: torch.Tensor,
        *caches: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        assert len(caches) == 2 * self.n_kv_heads * self.n_layers

        new_keys = []
        new_values = []
        for i, layer in enumerate(self.layers):
            key_t_caches = caches[: self.n_kv_caches][i * self.n_kv_heads : (i + 1) * self.n_kv_heads]
            value_caches = caches[self.n_kv_caches :][i * self.n_kv_heads : (i + 1) * self.n_kv_heads]

            x, keys, values = layer(x, key_t_caches, value_caches, attn_bias, (rope_embed_cos, rope_embed_sin))

            new_keys.extend(keys)
            new_values.extend(values)

        return x, *new_keys, *new_values


class LlamaInputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()

        self.embedding = skip_init(
            nn.Embedding, num_embeddings=vocab_size, embedding_dim=embed_dim, dtype=torch.float32, device=device
        )

    def load_weights(self, loader: ModelLoader):
        loader.load(self.embedding, "model.embed_tokens.weight")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class LlamaOutputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, rms_norm_eps: float):
        super().__init__()

        self.norm = LlamaRMSNorm(embed_dim=embed_dim, eps=rms_norm_eps, device=device)
        self.output_proj = skip_init(
            nn.Linear, in_features=embed_dim, out_features=vocab_size, bias=False, dtype=torch.float32, device=device
        )

        self.saved_samples: List[Sample] = []

    def load_weights(self, loader: ModelLoader):
        loader.load(self.norm, "model.norm.weight")

        # Models use tie embedding do not have lm_head.weight
        if loader.contain("lm_head.weight"):
            loader.load(self.output_proj, "lm_head.weight")
        else:
            loader.load(self.output_proj, "model.embed_tokens.weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    @property
    def input_names(self) -> List[str]:
        return ["x"]

    @property
    def output_names(self) -> List[str]:
        return ["logits"]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["x", "logits"]


class LlamaModel(nn.Module):
    """Wrapper for Llama model"""

    def __init__(
        self,
        model_folder: Path,
        model_params: ModelParams,
        graph_params: GraphParams,
        model_chunks: List[ExportableModule],
    ) -> None:
        super().__init__()
        self.model_folder = model_folder
        self.model_params = model_params
        self.graph_params = graph_params

        self.loader = ModelLoader(model_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(model_folder)

        self.input_embedding = LlamaInputEmbedding(vocab_size=model_params.vocab_size, embed_dim=model_params.embed_dim)

        self.output_embedding = LlamaOutputEmbedding(
            vocab_size=model_params.vocab_size, embed_dim=model_params.embed_dim, rms_norm_eps=model_params.rms_norm_eps
        )

        self.model_chunks: List[LlamaModelChunk] = nn.ModuleList(model_chunks)

        self.system_prompt_length = 0
        self.first_prompt = True
        self.last_logits: Optional[torch.Tensor] = None
        self.logits: List[float] = []

    def load_weights(self):
        self.input_embedding.load_weights(self.loader)
        self.output_embedding.load_weights(self.loader)
        for model_chunk in self.model_chunks:
            model_chunk.load_weights(self.loader)

        for param in self.parameters():
            param.requires_grad = False

    @property
    def kv_cache_position(self) -> int:
        return self.model_chunks[0].kv_cache_position

    def tokenize(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt").flatten()

    def get_attention_bias(self, batch_size: int) -> torch.Tensor:
        # Unmask tokens in KV caches
        attn_bias = torch.full(
            size=(batch_size, self.graph_params.context_size),
            fill_value=self.model_params.attention_mask_value,
            dtype=torch.float32,
            device=device,
        )
        attn_bias[:, : self.kv_cache_position] = 0

        # Causal mask
        cache_size = self.graph_params.cache_size
        attn_bias[:, cache_size : cache_size + batch_size] = torch.full(
            size=(batch_size, batch_size),
            fill_value=self.model_params.attention_mask_value,
            dtype=torch.float32,
            device=device,
        ).triu(diagonal=1)

        return attn_bias

    def update_logits(self, input_ids: torch.Tensor, logits: torch.Tensor):
        logits = logits.log_softmax(dim=-1)

        if self.last_logits is not None:
            self.logits.append(self.last_logits[input_ids[0]].item())

        self.last_logits = logits[-1]

        logits = logits[:-1]
        input_ids = input_ids[1:].reshape(-1, 1)
        self.logits.extend(logits.gather(dim=1, index=input_ids).flatten().tolist())

    @property
    def perplexity(self) -> float:
        return torch.exp(-torch.tensor(self.logits).mean()).item()

    @torch.no_grad()
    def eval_batch(
        self, input_ids: torch.Tensor, save_samples: bool = False, save_kv: bool = False, update_logits: bool = False
    ):
        x = self.input_embedding(input_ids)

        (batch_size,) = input_ids.shape
        last_outputs = (x,)
        attn_bias = self.get_attention_bias(batch_size)
        rope_embeds = LlamaRoPE.compute_embeds(
            dim=self.model_params.head_dim,
            start_position=self.kv_cache_position,
            end_position=self.kv_cache_position + batch_size,
            theta=self.model_params.rope_theta,
            device=device,
        )

        for model_chunk in self.model_chunks:
            inputs = model_chunk.get_inputs(last_outputs, attn_bias, rope_embeds)
            outputs = model_chunk(*inputs)

            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            if save_samples:
                model_chunk.saved_samples.append(Sample(inputs, outputs))

            if isinstance(model_chunk, KVCache):
                kv = outputs[-2 * model_chunk.n_kv_caches :]
                model_chunk.update_kv_caches(kv)

                if save_kv:
                    assert model_chunk.saved_kv is None
                    model_chunk.saved_kv = kv

            last_outputs = outputs

        x = last_outputs[0]
        logits = self.output_embedding(x)
        if save_samples:
            self.output_embedding.saved_samples.append(Sample((x,), (logits,)))

        if update_logits:
            # x = last_outputs[0]
            # logits = self.output_embedding(x)
            self.update_logits(input_ids, logits)

    def eval_system_prompt(self, prompt: str):
        assert self.first_prompt
        self.first_prompt = False

        input_ids = self.tokenize(prompt)
        self.system_prompt_length = len(input_ids)
        print(f"System prompt: {len(input_ids)} tokens: {input_ids.tolist()} {repr(self.tokenizer.decode(input_ids))}")

        self.eval_batch(input_ids, save_kv=True)

    def eval_prompt(self, prompt: str, batch_size: int, save_samples: bool = True, max_n_tokens: Optional[int] = None):
        input_ids = self.tokenize(prompt, add_special_tokens=self.first_prompt)

        self.first_prompt = False

        n_tokens = min(input_ids.nelement(), self.graph_params.cache_size - self.kv_cache_position, max_n_tokens)
        n_tokens -= n_tokens % batch_size
        print(f"Prompt: {len(input_ids)} tokens, truncated to {n_tokens} tokens")

        for i in tqdm(range(0, n_tokens, batch_size)):
            self.eval_batch(input_ids[i : i + batch_size], save_samples=save_samples, update_logits=True)

    def reset(self):
        self.first_prompt = self.system_prompt_length == 0
        self.last_logits = None
        self.logits = []

        for model_chunk in self.model_chunks:
            model_chunk.reset_kv_cache_position(self.system_prompt_length)

    def dump_config_template(self) -> dict:
        return {
            "model_parameters": {
                "n_layers": self.model_params.n_layers,
                "vocab_size": self.model_params.vocab_size,
                "embed_dim": self.model_params.embed_dim,
                "ffn_hidden_dim": self.model_params.ffn_hidden_dim,
                "head_dim": self.model_params.head_dim,
                "n_kv_heads": self.model_params.n_kv_heads,
                "rope_theta": self.model_params.rope_theta,
                "rms_norm_eps": self.model_params.rms_norm_eps,
                "attention_mask_value": self.model_params.attention_mask_value,
                "tie_embedding": self.model_params.tie_embedding,
            },
            "qnn_parameters": {"n_hvx_threads": 4},
            "graphs": [
                {
                    "type": model_chunk.module_type,
                    "start_layer_id": model_chunk.start_layer_id,
                    "end_layer_id": model_chunk.end_layer_id,
                    "batch_size": self.graph_params.batch_size,
                    "cache_size": self.graph_params.cache_size,
                    "context_size": self.graph_params.context_size,
                    # To be filled later
                    "graph_name": "",
                    "model_path": f"{args.model_name}_{i}.bin",
                    "kv_path_format": f"kv/layer_{{layer_id}}_{{kv_type}}_{{head_id}}.raw",
                    "kv_size": self.system_prompt_length,
                    "x_name": "x",
                    "out_name": "out",
                }
                for i, model_chunk in enumerate(self.model_chunks)
            ],
            "embeddings": [{
                "graph_name": "",
                "model_path": "lm_head.bin",
                "batch_size": self.graph_params.batch_size,
                "x_name": "x",
                "out_name": "logits",
            }],
        }


class OutputEmbeddingExporter:
    """Export a model chunk to ONNX model, quantization calibration data and configurations"""

    def __init__(
        self,
        graph_name: str,
        model_chunk: Union[LlamaOutputEmbedding, LlamaInputEmbedding],
        use_fp16: bool,
        output_folder: Path,
    ):
        self.graph_name = graph_name
        self.model_chunk = model_chunk
        self.use_fp16 = use_fp16
        self.output_folder = output_folder

    @torch.no_grad()
    def export_onnx_model(self):
        onnx_model_folder = self.output_folder / "onnx_model"
        onnx_model_folder.mkdir(parents=True, exist_ok=True)

        onnx_model_path = onnx_model_folder / f"{self.graph_name}.onnx"
        torch.onnx.export(
            model=self.model_chunk,
            args=self.model_chunk.saved_samples[0].inputs,
            f=str(onnx_model_path),
            input_names=self.model_chunk.input_names,
            output_names=self.model_chunk.output_names,
        )

        onnx_model = onnx.load(onnx_model_path, load_external_data=False)
        self.onnx_model = shape_inference.infer_shapes(onnx_model)

    def export_io_spec(self):
        def dump_info_list(io_type: Literal["in", "out"], names: List[str], tensors: List[torch.Tensor]) -> List[dict]:
            return [
                {
                    "name": name,
                    "type": io_type,
                    "dtype": "int64" if name == "input_ids" else "float32",
                    "preserve_dtype": name in self.model_chunk.dtype_preserved_io_names,
                    "shape": list(tensor.shape),
                }
                for name, tensor in zip(names, tensors)
            ]

        io_spec = [
            *dump_info_list("in", self.model_chunk.input_names, self.model_chunk.saved_samples[0].inputs),
            *dump_info_list("out", self.model_chunk.output_names, self.model_chunk.saved_samples[0].outputs),
        ]

        export_json(io_spec, self.output_folder / f"{self.graph_name}.io.json")

    def export_quantization_config(self):
        class Encoding(NamedTuple):
            category: Literal["activation", "param"]
            bitwidth: int
            dtype: Literal["float", "int"]

        encoding_map: Dict[str, Encoding] = {}

        def update_encoding(name: str, encoding: Encoding):
            """Set encoding for name. Only update if the bitwidth is larger than the previous setting."""

            if name not in encoding_map or encoding.bitwidth > encoding_map[name].bitwidth:
                encoding_map[name] = encoding

        def encode_activation(node: str | onnx.ValueInfoProto, bitwidth: int):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("activation", bitwidth, "float"))

        def encode_output(node: onnx.NodeProto, bitwidth: int):
            for name in node.output:
                update_encoding(name, Encoding("activation", bitwidth, "float"))

        def encode_param(node: str | onnx.NodeProto, bitwidth: Encoding, dtype: Literal["float", "int"]):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("param", bitwidth, dtype))

        def match(target: str | onnx.NodeProto | onnx.TensorProto | onnx.ValueInfoProto, pattern: str):
            if not isinstance(target, str):
                target = target.name
            return re.fullmatch(pattern, target) is not None

        graph = self.onnx_model.graph

        # Inputs
        encode_activation("x", 32)
        encode_activation("attn_bias", 16)

        # KV cache: FP16
        for node in graph.input:
            if match(node, "layer_[0-9]+_(key_t|value)_cache_[0-9]+"):
                encode_activation(node, 16)
        for node in graph.output:
            if match(node, "layer_[0-9]+_(key|value)_[0-9]+"):
                encode_activation(node, 16)

        # Attention core: FP16
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/attn/core.*"):
                encode_output(node, 16)

        # Residual connection: FP32
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/(attn|ffn|ffn/down_proj)/Add(_[0-9]+|)"):
                encode_output(node, 32)

        # RMSNorm: FP32
        for node in graph.initializer:
            if match(node, "layers\.[0-9]+\.(attn|ffn)\.norm\.weight"):
                encode_param(node, 32, "float")
        for node in graph.node:
            # print(node)
            if match(node, "/layers\.[0-9]+/(attn|ffn)/norm.*"):
                encode_output(node, 32)
            elif match(node, "/norm.*"):
                encode_output(node, 32)

        if self.use_fp16:
            print("NOTE: Use FP16 for output embedding.")
            for node in graph.initializer:
                encode_param(node, 16, "float")
            for node in graph.node:
                encode_output(node, 16)

        # Generate config
        config = {
            "version": "0.6.1",
            "quantizer_args": {
                "activation_bitwidth": 16,
                "param_bitwidth": 4,
                "dtype": "int",
                "per_channel_quantization": True,
                "quant_scheme": "post_training_tf",
            },
            "activation_encodings": {},
            "param_encodings": {},
        }

        for name, encoding in sorted(encoding_map.items(), key=lambda item: item[0]):
            config[f"{encoding.category}_encodings"][name] = [{
                "bitwidth": encoding.bitwidth,
                "dtype": encoding.dtype,
                "is_symmetric": str(encoding.category == "param"),
            }]

        export_json(config, self.output_folder / f"{self.graph_name}.encodings")

    def export_sample_inputs(self):
        input_list = []
        for i, samples in enumerate(self.model_chunk.saved_samples):
            data_folder = self.output_folder / "data" / str(i)
            data_folder.mkdir(parents=True, exist_ok=True)

            tensor_paths = []
            for name, tensor in zip(self.model_chunk.input_names, samples.inputs, strict=True):
                output_path = data_folder / f"{name}.raw"
                tensor.cpu().numpy().tofile(output_path)
                tensor_paths.append(f"{name}:={output_path}")

            input_list.append(" ".join(tensor_paths))

        with open(self.output_folder / "input_list.txt", "w") as f:
            f.write("\n".join(input_list))

    def export_saved_kv(self):
        kv_folder = self.output_folder / "kv"
        kv_folder.mkdir(parents=True, exist_ok=True)

        for name, tensor in zip(self.model_chunk.kv_names, self.model_chunk.saved_kv, strict=True):
            tensor.cpu().numpy().tofile(kv_folder / f"{name}.raw")

    def export(self):
        self.export_onnx_model()
        self.export_io_spec()
        self.export_quantization_config()
        self.export_sample_inputs()
        if isinstance(self.model_chunk, KVCache) and self.model_chunk.saved_kv is not None:
            self.export_saved_kv()


class ModelChunkExporter:
    """Export a model chunk to ONNX model, quantization calibration data and configurations"""

    def __init__(
        self, graph_name: str, model_chunk: LlamaModelChunk, fp16_overrides: Dict[str, List[int]], output_folder: Path
    ):
        self.graph_name = graph_name
        self.model_chunk = model_chunk
        self.fp16_overrides = fp16_overrides
        self.output_folder = output_folder

    @torch.no_grad()
    def export_onnx_model(self):
        onnx_model_folder = self.output_folder / "onnx_model"
        onnx_model_folder.mkdir(parents=True, exist_ok=True)

        onnx_model_path = onnx_model_folder / f"{self.graph_name}.onnx"
        torch.onnx.export(
            model=self.model_chunk,
            args=self.model_chunk.saved_samples[0].inputs,
            f=str(onnx_model_path),
            input_names=self.model_chunk.input_names,
            output_names=self.model_chunk.output_names,
        )

        onnx_model = onnx.load(onnx_model_path, load_external_data=False)
        self.onnx_model = shape_inference.infer_shapes(onnx_model)

    def export_io_spec(self):
        def dump_info_list(io_type: Literal["in", "out"], names: List[str], tensors: List[torch.Tensor]) -> List[dict]:
            return [
                {
                    "name": name,
                    "type": io_type,
                    "dtype": "float32",
                    "preserve_dtype": name in self.model_chunk.dtype_preserved_io_names,
                    "shape": list(tensor.shape),
                }
                for name, tensor in zip(names, tensors)
            ]

        io_spec = [
            *dump_info_list("in", self.model_chunk.input_names, self.model_chunk.saved_samples[0].inputs),
            *dump_info_list("out", self.model_chunk.output_names, self.model_chunk.saved_samples[0].outputs),
        ]

        export_json(io_spec, self.output_folder / f"{self.graph_name}.io.json")

    def export_quantization_config(self):
        class Encoding(NamedTuple):
            category: Literal["activation", "param"]
            bitwidth: int
            dtype: Literal["float", "int"]

        encoding_map: Dict[str, Encoding] = {}

        def update_encoding(name: str, encoding: Encoding):
            """Set encoding for name. Only update if the bitwidth is larger than the previous setting."""

            if name not in encoding_map or encoding.bitwidth > encoding_map[name].bitwidth:
                encoding_map[name] = encoding

        def encode_activation(node: Union[str, onnx.ValueInfoProto], bitwidth: int):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("activation", bitwidth, "float"))

        def encode_output(node: onnx.NodeProto, bitwidth: int):
            for name in node.output:
                update_encoding(name, Encoding("activation", bitwidth, "float"))

        def encode_param(node: Union[str, onnx.NodeProto], bitwidth: Encoding, dtype: Literal["float", "int"]):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("param", bitwidth, dtype))

        def match(target: Union[str, onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto], pattern: str):
            if not isinstance(target, str):
                target = target.name
            return re.fullmatch(pattern, target) is not None

        graph = self.onnx_model.graph

        # Inputs
        encode_activation("x", 32)
        encode_activation("attn_bias", 16)
        encode_activation("rope_embed_cos", 16)
        encode_activation("rope_embed_sin", 16)

        # KV cache: FP16
        for node in graph.input:
            if match(node, "layer_[0-9]+_(key_t|value)_cache_[0-9]+"):
                encode_activation(node, 16)
        for node in graph.output:
            if match(node, "layer_[0-9]+_(key|value)_[0-9]+"):
                encode_activation(node, 16)

        # Attention core: FP16
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/attn/core.*"):
                encode_output(node, 16)

        # Manually specified FP16 attention/FFN layers
        for layer_type in ["attn", "ffn"]:
            layer_id_list = self.fp16_overrides[layer_type]

            for layer_id in layer_id_list:
                if not (self.model_chunk.start_layer_id <= layer_id < self.model_chunk.end_layer_id):
                    continue

                # NOTE: Layer ids in an ONNX model are always started from 0
                count = 0
                index = layer_id - self.model_chunk.start_layer_id
                for node in graph.initializer:
                    if match(node, f"layers\.{index}\.{layer_type}.*"):
                        count += 1
                        encode_param(node, 16, "float")
                for node in graph.node:
                    if match(node, f"/layers\.{index}/{layer_type}.*"):
                        count += 1
                        encode_output(node, 16)
                print(f'Override {count} nodes in layer "{layer_type}_{layer_id}" to FP16')

        if self.fp16_overrides["rope"] == True:
            print("Use FP16 for attention RoPE.")
            for node in graph.node:
                if "/attn/rope" in node.name:
                    encode_output(node, 16)

        # Residual connection: FP32
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/(attn|ffn|ffn/down_proj)/Add(_[0-9]+|)"):
                encode_output(node, 32)

        # RMSNorm: FP32
        for node in graph.initializer:
            if match(node, "layers\.[0-9]+\.(attn|ffn)\.norm\.weight"):
                encode_param(node, 32, "float")
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/(attn|ffn)/norm.*"):
                encode_output(node, 32)

        # FP16 components
        for node in graph.initializer:
            if "fp16_" in node.name:
                encode_param(node, 16, "float")
        for node in graph.node:
            if "fp16_" in node.name:
                encode_output(node, 16)

        if self.fp16_overrides["qkv_heads"] == True:
            print("Use FP16 for QKV heads.")
            for node in graph.initializer:
                if match(node, ".*[qkv]_heads.*"):
                    encode_param(node, 16, "float")
            for node in graph.node:
                if match(node, ".*[qkv]_heads.*"):
                    encode_output(node, 16)

        # Generate config
        config = {
            "version": "0.6.1",
            "quantizer_args": {
                "activation_bitwidth": 16,
                "param_bitwidth": 4,
                "dtype": "int",
                "per_channel_quantization": True,
                "quant_scheme": "post_training_tf",
            },
            "activation_encodings": {},
            "param_encodings": {},
        }

        for name, encoding in sorted(encoding_map.items(), key=lambda item: item[0]):
            config[f"{encoding.category}_encodings"][name] = [{
                "bitwidth": encoding.bitwidth,
                "dtype": encoding.dtype,
                "is_symmetric": str(encoding.category == "param"),
            }]

        export_json(config, self.output_folder / f"{self.graph_name}.encodings")

    def export_sample_inputs(self):
        input_list = []
        for i, samples in enumerate(self.model_chunk.saved_samples):
            data_folder = (self.output_folder / "data" / str(i)).resolve()
            data_folder.mkdir(parents=True, exist_ok=True)

            tensor_paths = []
            for name, tensor in zip(self.model_chunk.input_names, samples.inputs):
                output_path = data_folder / f"{name}.raw"
                tensor.cpu().numpy().tofile(output_path)
                tensor_paths.append(f"{name}:={output_path}")

            input_list.append(" ".join(tensor_paths))

        with open(self.output_folder / "input_list.txt", "w") as f:
            f.write("\n".join(input_list))

    def export_saved_kv(self):
        kv_folder = self.output_folder / "kv"
        kv_folder.mkdir(parents=True, exist_ok=True)

        for name, tensor in zip(self.model_chunk.kv_names, self.model_chunk.saved_kv):
            tensor.cpu().numpy().tofile(kv_folder / f"{name}.raw")

    def export(self):
        self.export_onnx_model()
        self.export_io_spec()
        self.export_quantization_config()
        self.export_sample_inputs()
        if isinstance(self.model_chunk, KVCache) and self.model_chunk.saved_kv is not None:
            self.export_saved_kv()


print("Creating model...")

model_params: ModelParams = model_map[args.model_name]()
graph_params: GraphParams = graph_map[args.graph_name]()

assert model_params.n_layers % args.n_model_chunks == 0
n_layers_per_model_chunk = model_params.n_layers // args.n_model_chunks

model_chunks = [
    LlamaModelChunk(
        start_layer_id=i,
        end_layer_id=(i + n_layers_per_model_chunk),
        embed_dim=model_params.embed_dim,
        n_heads=model_params.n_heads,
        n_kv_heads=model_params.n_kv_heads,
        context_size=graph_params.context_size,
        ffn_hidden_dim=model_params.ffn_hidden_dim,
        rms_norm_eps=model_params.rms_norm_eps,
        has_qkv_bias=model_params.has_qkv_bias,
        use_drelu=model_params.use_drelu,
        cache_size=graph_params.cache_size,
        n_fp16_heads=model_params.n_fp16_heads,
        n_fp16_neurons=model_params.n_fp16_neurons,
        stat_folder=args.model_folder / "stat",
    )
    for i in range(0, model_params.n_layers, n_layers_per_model_chunk)
]


model = LlamaModel(
    model_folder=args.model_folder, model_params=model_params, graph_params=graph_params, model_chunks=model_chunks
)

print("Loading model weights...")
model.load_weights()

if args.system_prompt_file is not None:
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()
    model.eval_system_prompt(system_prompt)

with open(args.prompt_file, "r") as f:
    prompt = f.read()


def eval_prompt(save_samples: bool = False):
    model.eval_prompt(
        prompt=prompt,
        batch_size=model.graph_params.batch_size,
        save_samples=save_samples,
        max_n_tokens=args.max_n_tokens,
    )
    print(f"Perplexity: {model.perplexity:.4f}")


eval_prompt(save_samples=True)


def dump_quant_debug_info(attr_name: str):
    attr_list = sorted(
        (getattr(module, attr_name), name)
        for name, module in model.named_modules()
        if isinstance(module, LinearWithQuantizationDebugger)
    )

    if len(attr_list) == 0:
        return

    debug_folder = Path("./debug")
    debug_folder.mkdir(exist_ok=True, parents=True)

    debug_file = debug_folder / f"{attr_name}.txt"
    with open(debug_file, "w") as f:
        for attr, name in attr_list:
            f.write(f"{name} {attr}\n")

    print(f'NOTE: Dumped "{debug_file}".')


dump_quant_debug_info("top_input_norms")
dump_quant_debug_info("top_output_norms")
dump_quant_debug_info("quant_cos_sim")


def dump_activation_stat():
    assert len(model_chunks) == 1

    model_chunk = model_chunks[0]

    n_layers = len(model_chunk.layers)

    def attn_stat(attn_module: LlamaAttention):
        head_dim = attn_module.head_dim
        act_abs_sum = attn_module.out_act_abs_sum.reshape(-1, head_dim).sum(dim=-1)
        return act_abs_sum.argsort(descending=True).tolist()

    def ffn_stat(ffn_module: LlamaFeedForwardChunk):
        return ffn_module.down_act_abs_sum.argsort(descending=True).tolist()

    stat_output_folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_layers):
        with open(stat_output_folder / f"attn_{i}_stat.json", "w") as f:
            json.dump(attn_stat(model_chunk.layers[i].attn), f)
        with open(stat_output_folder / f"ffn_{i}_stat.json", "w") as f:
            json.dump(ffn_stat(model_chunk.layers[i].ffn.int4_chunk), f)


if stat_output_folder is not None:
    dump_activation_stat()

if args.output_folder is None:
    exit(0)

args.output_folder.mkdir(parents=True, exist_ok=True)

config_template = model.dump_config_template()
for graph_info in config_template["graphs"]:
    graph_info["graph_name"] = args.graph_name
for embedding_info in config_template["embeddings"]:
    embedding_info["graph_name"] = args.graph_name
export_json(config_template, args.output_folder / f"config_{args.graph_name}.json")

for i, model_chunk in enumerate(model.model_chunks):
    print(f'Exporting "model_chunk_{i}"...')

    output_folder = args.output_folder / f"model_chunk_{i}" / args.graph_name
    output_folder.mkdir(parents=True, exist_ok=True)

    exporter = ModelChunkExporter(
        graph_name=args.graph_name,
        model_chunk=model_chunk,
        fp16_overrides={
            "attn": model_params.fp16_attention_layers,
            "ffn": model_params.fp16_ffn_layers,
            "rope": model_params.fp16_rope,
            "qkv_heads": model_params.fp16_qkv_heads,
        },
        output_folder=output_folder,
    )
    exporter.export()

print("Exporting output embedding...")
output_folder = args.output_folder / "output_embedding" / args.graph_name
output_folder.mkdir(parents=True, exist_ok=True)

exporter = OutputEmbeddingExporter(
    graph_name=args.graph_name,
    model_chunk=model.output_embedding,
    use_fp16=args.fp16_lm_head,
    output_folder=output_folder,
)
exporter.export()
