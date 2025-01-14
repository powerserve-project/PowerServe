import math
from pathlib import Path

import safetensors
import torch
from fake_quantize import Quantizer
from torch import Tensor, nn
from torch.nn.utils import skip_init


class ModelLoader:
    """Helper class to load weight tensors from safetensors"""

    def __init__(self, folder: Path):
        """Load tensors from safetensor files and create a mapping between tensor names and tensors"""

        self.tensor_map: dict[str, torch.Tensor] = {}
        for model_shard_file in folder.glob("*.safetensors"):
            tensors = safetensors.safe_open(model_shard_file, "pt")
            for name in tensors.keys():
                self.tensor_map[name] = tensors.get_tensor(name)

    def contain(self, name: str) -> bool:
        return name in self.tensor_map

    def load(self, dest: nn.Module | torch.Tensor, name: str, transposed: bool = False):
        """Look up tensor in tensor map and copy data to destination tensor"""

        tensor = self.tensor_map[name]

        target = None
        if isinstance(dest, nn.Module):
            target = dest.weight.data
        elif isinstance(dest, torch.Tensor):
            target = dest.data
        else:
            raise RuntimeError

        if transposed:
            tensor = tensor.T

        assert target.shape == tensor.shape, f"Expect {tuple(target.shape)}, got {tuple(tensor.shape)}"
        target.copy_(tensor.to(torch.float32))


class LinearWithQuantizationDebugger(nn.Linear):
    max_n_records = 128

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.first_forward = True
        self.input_norm_records = []
        self.output_norm_records = []
        self.cos_sim_records = []

    @property
    def top_input_norms(self) -> list[tuple[float, int]]:
        records = [(-v, i) for i, v in enumerate(self.input_norm_records)]
        records = sorted(records)[:10]
        return [(round(-v, 5), i) for v, i in records]

    @property
    def top_output_norms(self) -> list[tuple[float, int]]:
        records = [(-v, i) for i, v in enumerate(self.output_norm_records)]
        records = sorted(records)[:10]
        return [(round(-v, 5), i) for v, i in records]

    @property
    def quant_cos_sim(self) -> float:
        records = self.cos_sim_records[-self.max_n_records :]
        return round(sum(records) / len(records), 5)

    def forward(self, x: Tensor) -> Tensor:
        if self.first_forward:
            self.quantized_weight = Quantizer(bitwidth=4, per_channel=True).quantize(self.weight)
            self.first_forward = False

        self.input_norm_records.extend(x.norm(dim=-1).tolist())
        out = super().forward(x)
        self.output_norm_records.extend(out.norm(dim=-1).tolist())

        quantized_out = nn.functional.linear(x, self.quantized_weight, self.bias)
        self.cos_sim_records.extend(torch.cosine_similarity(out, quantized_out, dim=-1).tolist())

        return out


# TODO: Add command line option for it.
linear_class = nn.Linear  # Disable quantization debug
# linear_class = LinearWithQuantizationDebugger

stat_output_folder = None
# Uncomment the following line to enable activation stat profiling
# stat_output_folder = Path("./smallthinker_3b_stat")


class LlamaRoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def compute_embeds(
        dim: int, start_position: int, end_position: int, theta: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert dim % 2 == 0
        inv_freq = 1 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(start=start_position, end=end_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq).to(device)  # (n_positions, dim / 2)
        return (freqs.cos(), freqs.sin())

    def forward(self, x: torch.Tensor, rope_embeds: tuple[torch.Tensor]) -> torch.Tensor:
        rope_cos = rope_embeds[0]  # (batch_size, dim / 2)
        rope_sin = rope_embeds[1]  # (batch_size, dim / 2)

        head_dim = x.shape[-1]
        x0 = x[:, : head_dim // 2]
        x1 = x[:, head_dim // 2 :]
        return torch.cat((x0 * rope_cos - x1 * rope_sin, x0 * rope_sin + x1 * rope_cos), dim=-1)


class LlamaRMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float, device: torch.device):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim, dtype=torch.float32, device=device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).reciprocal()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x) * self.weight


class LlamaAttentionCore(nn.Module):
    def __init__(self, n_kv_heads: int, group_size: int, context_size: int, device: torch.device):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.group_size = group_size
        self.context_size = context_size
        self.neg_inf = torch.tensor([-1e4], dtype=torch.float32, device=device)

    def forward(
        self,
        queries: list[torch.Tensor],
        keys: list[torch.Tensor],
        key_t_caches: list[torch.Tensor],
        values: list[torch.Tensor],
        value_caches: list[torch.Tensor],
        attn_bias: torch.Tensor,
        kq_scale: float,
    ):
        n_heads = len(queries)
        assert len(keys) == self.n_kv_heads
        assert self.n_kv_heads * self.group_size == n_heads

        batch_size = values[0].shape[0]
        cache_size = value_caches[0].shape[0]
        kv_pad_size = self.context_size - batch_size - cache_size

        scaled_keys = []
        scaled_values = []
        head_outs = []
        for i in range(self.n_kv_heads):
            scaled_key = keys[i] * kq_scale
            scaled_value = torch.max(values[i], self.neg_inf)  # No scaling
            scaled_keys.append(scaled_key)
            scaled_values.append(scaled_value)

            scaled_key_t = scaled_key.transpose(0, 1)
            padded_key_t = nn.functional.pad(scaled_key_t, (0, kv_pad_size))
            padded_value = nn.functional.pad(scaled_value, (0, 0, 0, kv_pad_size))

            all_keys_t = torch.cat((key_t_caches[i], padded_key_t), dim=-1)
            all_values = torch.cat((value_caches[i], padded_value), dim=-2)

            for query in queries[i * self.group_size : (i + 1) * self.group_size]:
                score = query @ all_keys_t
                score = score + attn_bias
                score = score.softmax(dim=-1)
                out = score @ all_values
                head_outs.append(out)

        return head_outs, scaled_keys, scaled_values


class LlamaAttentionOutputProj(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, n_heads: int, device: torch.device):
        super().__init__()

        self.output_proj = skip_init(
            linear_class,
            in_features=head_dim * n_heads,
            out_features=embed_dim,
            bias=False,
            dtype=torch.float32,
            device=device,
        )

    def forward(self, head_inputs: list[Tensor]) -> Tensor:
        x = torch.cat(head_inputs, dim=-1)
        return self.output_proj(x)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        context_size: int,
        has_qkv_bias: bool,
        rms_norm_eps: float,
        fp16_head_ids: list[int],
        device: torch.device,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.has_qkv_bias = has_qkv_bias
        self.device = device

        self.fp16_head_ids = sorted(fp16_head_ids)
        self.int4_head_ids = sorted(set(range(n_heads)).difference(fp16_head_ids))

        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.head_dim = embed_dim // n_heads
        self.group_size = n_heads // n_kv_heads

        self.norm = LlamaRMSNorm(embed_dim=embed_dim, eps=rms_norm_eps, device=device)
        self.rope = LlamaRoPE()
        self.core = LlamaAttentionCore(
            n_kv_heads=n_kv_heads, group_size=self.group_size, context_size=context_size, device=device
        )

        self.fp16_q_heads = nn.ModuleList()
        self.int4_q_heads = nn.ModuleList()
        self.q_heads = []
        for i in range(n_heads):
            q_head = skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.head_dim,
                bias=has_qkv_bias,
                dtype=torch.float32,
                device=device,
            )

            if i in self.fp16_head_ids:
                self.fp16_q_heads.append(q_head)
            else:
                self.int4_q_heads.append(q_head)

            self.q_heads.append(q_head)

        self.k_heads = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.head_dim,
                bias=has_qkv_bias,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_kv_heads)
        ])
        self.v_heads = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.head_dim,
                bias=has_qkv_bias,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_kv_heads)
        ])

        if len(self.fp16_head_ids) > 0:
            self.fp16_o_proj = LlamaAttentionOutputProj(
                embed_dim=embed_dim, head_dim=self.head_dim, n_heads=len(self.fp16_head_ids), device=device
            )
        if len(self.int4_head_ids) > 0:
            self.int4_o_proj = LlamaAttentionOutputProj(
                embed_dim=embed_dim, head_dim=self.head_dim, n_heads=len(self.int4_head_ids), device=device
            )

        if stat_output_folder is not None:
            self.out_act_abs_sum = torch.zeros((embed_dim,), dtype=torch.float32)

    def load_weights(self, loader: ModelLoader):
        loader.load(self.norm, f"model.layers.{self.layer_id}.input_layernorm.weight")

        wq = torch.empty(self.embed_dim, self.embed_dim, dtype=torch.float32, device=self.device)
        loader.load(wq, f"model.layers.{self.layer_id}.self_attn.q_proj.weight")
        if self.has_qkv_bias:
            bq = torch.empty(self.embed_dim, dtype=torch.float32, device=self.device)
            loader.load(bq, f"model.layers.{self.layer_id}.self_attn.q_proj.bias")

        for i in range(self.n_heads):
            self.q_heads[i].weight.data.copy_(wq[i * self.head_dim : (i + 1) * self.head_dim, :])
            if self.has_qkv_bias:
                self.q_heads[i].bias.data.copy_(bq[i * self.head_dim : (i + 1) * self.head_dim])

        wk = torch.empty(self.head_dim * self.n_kv_heads, self.embed_dim, dtype=torch.float32, device=self.device)
        loader.load(wk, f"model.layers.{self.layer_id}.self_attn.k_proj.weight")
        if self.has_qkv_bias:
            bk = torch.empty(self.head_dim * self.n_kv_heads, dtype=torch.float32, device=self.device)
            loader.load(bk, f"model.layers.{self.layer_id}.self_attn.k_proj.bias")

        for i in range(self.n_kv_heads):
            self.k_heads[i].weight.data.copy_(wk[i * self.head_dim : (i + 1) * self.head_dim, :])
            if self.has_qkv_bias:
                self.k_heads[i].bias.data.copy_(bk[i * self.head_dim : (i + 1) * self.head_dim])

        wv = torch.empty(self.head_dim * self.n_kv_heads, self.embed_dim, dtype=torch.float32, device=self.device)
        loader.load(wv, f"model.layers.{self.layer_id}.self_attn.v_proj.weight")
        if self.has_qkv_bias:
            bv = torch.empty(self.head_dim * self.n_kv_heads, dtype=torch.float32, device=self.device)
            loader.load(bv, f"model.layers.{self.layer_id}.self_attn.v_proj.bias")

        for i in range(self.n_kv_heads):
            self.v_heads[i].weight.data.copy_(wv[i * self.head_dim : (i + 1) * self.head_dim, :])
            if self.has_qkv_bias:
                self.v_heads[i].bias.data.copy_(bv[i * self.head_dim : (i + 1) * self.head_dim])

        wo = torch.empty(self.embed_dim, self.embed_dim, dtype=torch.float32, device=self.device)
        loader.load(wo, f"model.layers.{self.layer_id}.self_attn.o_proj.weight")
        o_heads = wo.reshape(self.embed_dim, self.n_heads, self.head_dim)
        if len(self.fp16_head_ids) > 0:
            self.fp16_o_proj.output_proj.weight.data.copy_(o_heads[:, self.fp16_head_ids].reshape(self.embed_dim, -1))
        if len(self.int4_head_ids) > 0:
            self.int4_o_proj.output_proj.weight.data.copy_(o_heads[:, self.int4_head_ids].reshape(self.embed_dim, -1))

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, embed_dim)
        key_t_caches: tuple[torch.Tensor],  # Transposed keys: (head_dim, cache_size) * n_kv_heads
        value_caches: tuple[torch.Tensor],  # (cache_size, head_dim) * n_kv_heads
        attn_bias: torch.Tensor,  # (batch_size, context_size)
        rope_embeds: tuple[torch.Tensor],  # (batch_size, head_dim / 2) * 2
    ) -> tuple[torch.Tensor, tuple[torch.Tensor], tuple[torch.Tensor]]:
        """Returns the attention output, keys and values of input x"""

        attn_input = self.norm(x)
        queries = [self.rope(q_head(attn_input), rope_embeds) for q_head in self.q_heads]
        keys = [self.rope(k_head(attn_input), rope_embeds) for k_head in self.k_heads]
        values = [v_head(attn_input) for v_head in self.v_heads]

        head_outs, scaled_keys, scaled_values = self.core(
            queries, keys, key_t_caches, values, value_caches, attn_bias, kq_scale=(1 / math.sqrt(self.head_dim))
        )

        fp16_inputs = [head_outs[i] for i in self.fp16_head_ids]
        int4_inputs = [head_outs[i] for i in self.int4_head_ids]

        if hasattr(self, "out_act_abs_sum"):
            assert len(fp16_inputs) == 0, "Set n_fp16_heads to zero first"
            out = torch.cat(int4_inputs, dim=-1)
            self.out_act_abs_sum += out.abs().sum(dim=0)

        outs = [x]
        if len(fp16_inputs) > 0:
            outs.append(self.fp16_o_proj(fp16_inputs))
        if len(int4_inputs) > 0:
            outs.append(self.int4_o_proj(int4_inputs))

        return sum(outs), tuple(scaled_keys), tuple(scaled_values)


class LlamaFeedForwardChunk(nn.Module):
    def __init__(self, embed_dim: int, ffn_hidden_dim: int, use_drelu: bool, device: torch.device):
        super().__init__()
        self.use_drelu = use_drelu

        if use_drelu:
            self.relu = nn.ReLU()
        else:
            self.silu = nn.SiLU()

        self.gate_proj = skip_init(
            linear_class,
            in_features=embed_dim,
            out_features=ffn_hidden_dim,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.up_proj = skip_init(
            linear_class,
            in_features=embed_dim,
            out_features=ffn_hidden_dim,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.down_proj = skip_init(
            linear_class,
            in_features=ffn_hidden_dim,
            out_features=embed_dim,
            bias=False,
            dtype=torch.float32,
            device=device,
        )

        if stat_output_folder is not None:
            self.down_act_abs_sum = torch.zeros((ffn_hidden_dim,), dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        if self.use_drelu:
            gate = self.relu(gate)
        else:
            gate = self.silu(gate)

        up = self.up_proj(x)
        if self.use_drelu:
            up = self.relu(up)

        out = gate * up

        if hasattr(self, "down_act_abs_sum"):
            self.down_act_abs_sum += out.abs().sum(dim=0)

        return self.down_proj(out)


class LlamaFeedForward(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embed_dim: int,
        ffn_hidden_dim: int,
        rms_norm_eps: float,
        use_drelu: bool,
        fp16_neuron_ids: list[int],
        device: torch.device,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.device = device

        self.fp16_neuron_ids = sorted(fp16_neuron_ids)
        self.int4_neuron_ids = sorted(set(range(ffn_hidden_dim)).difference(fp16_neuron_ids))

        self.norm = LlamaRMSNorm(embed_dim=embed_dim, eps=rms_norm_eps, device=device)

        if len(self.fp16_neuron_ids) > 0:
            self.fp16_chunk = LlamaFeedForwardChunk(
                embed_dim=embed_dim, ffn_hidden_dim=len(self.fp16_neuron_ids), use_drelu=use_drelu, device=device
            )

        if len(self.int4_neuron_ids) > 0:
            self.int4_chunk = LlamaFeedForwardChunk(
                embed_dim=embed_dim, ffn_hidden_dim=len(self.int4_neuron_ids), use_drelu=use_drelu, device=device
            )

        if stat_output_folder is not None:
            assert len(self.fp16_neuron_ids) == 0, "Set n_fp16_neurons to zero first"

    def load_weights(self, loader: ModelLoader):
        loader.load(self.norm, f"model.layers.{self.layer_id}.post_attention_layernorm.weight")

        gate_proj = torch.empty((self.ffn_hidden_dim, self.embed_dim), dtype=torch.float32, device=self.device)
        up_proj = torch.empty((self.ffn_hidden_dim, self.embed_dim), dtype=torch.float32, device=self.device)
        down_proj = torch.empty((self.embed_dim, self.ffn_hidden_dim), dtype=torch.float32, device=self.device)

        loader.load(gate_proj, f"model.layers.{self.layer_id}.mlp.gate_proj.weight")
        loader.load(up_proj, f"model.layers.{self.layer_id}.mlp.up_proj.weight")
        loader.load(down_proj, f"model.layers.{self.layer_id}.mlp.down_proj.weight")

        if len(self.fp16_neuron_ids) > 0:
            self.fp16_chunk.gate_proj.weight.data.copy_(gate_proj[self.fp16_neuron_ids, :])
            self.fp16_chunk.up_proj.weight.data.copy_(up_proj[self.fp16_neuron_ids, :])
            self.fp16_chunk.down_proj.weight.data.copy_(down_proj[:, self.fp16_neuron_ids])

        if len(self.int4_neuron_ids) > 0:
            self.int4_chunk.gate_proj.weight.data.copy_(gate_proj[self.int4_neuron_ids, :])
            self.int4_chunk.up_proj.weight.data.copy_(up_proj[self.int4_neuron_ids, :])
            self.int4_chunk.down_proj.weight.data.copy_(down_proj[:, self.int4_neuron_ids])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffn_input = self.norm(x)

        outs = [x]
        if len(self.fp16_neuron_ids) > 0:
            outs.append(self.fp16_chunk(ffn_input))
        if len(self.int4_neuron_ids) > 0:
            outs.append(self.int4_chunk(ffn_input))

        return sum(outs)


class LlamaTransformer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        context_size: int,
        ffn_hidden_dim: int,
        rms_norm_eps: float,
        has_qkv_bias: bool,
        use_drelu: bool,
        fp16_head_ids: list[int],
        fp16_neuron_ids: list[int],
        device: torch.device,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.attn = LlamaAttention(
            layer_id=layer_id,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            context_size=context_size,
            has_qkv_bias=has_qkv_bias,
            rms_norm_eps=rms_norm_eps,
            fp16_head_ids=fp16_head_ids,
            device=device,
        )
        self.ffn = LlamaFeedForward(
            layer_id=layer_id,
            embed_dim=embed_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            rms_norm_eps=rms_norm_eps,
            use_drelu=use_drelu,
            fp16_neuron_ids=fp16_neuron_ids,
            device=device,
        )

    def load_weights(self, loader: ModelLoader):
        self.attn.load_weights(loader)
        self.ffn.load_weights(loader)

    def forward(
        self,
        x: torch.Tensor,
        key_t_caches: tuple[torch.Tensor],
        value_caches: tuple[torch.Tensor],
        attn_bias: torch.Tensor,
        rope_embeds: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor], tuple[torch.Tensor]]:
        attn_out, keys, values = self.attn(x, key_t_caches, value_caches, attn_bias, rope_embeds)
        ffn_out = self.ffn(attn_out)
        return ffn_out, keys, values
