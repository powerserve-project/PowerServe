class ModelParams:
    has_qkv_bias: bool
    use_drelu: bool
    tie_embedding: bool

    n_layers: int
    vocab_size: int
    ffn_hidden_dim: int
    head_dim: int
    n_heads: int
    n_kv_heads: int

    rope_theta: float
    rms_norm_eps: float
    attention_mask_value: float

    fp16_attention_layers: list[int]
    fp16_ffn_layers: list[int]
    fp16_rope: bool
    fp16_qkv_heads: bool
    n_fp16_heads: int | dict[int, int]  # Key is layer id
    n_fp16_neurons: int | dict[int, int]

    @property
    def embed_dim(self) -> int:
        return self.head_dim * self.n_heads

    @property
    def group_size(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads


class Llama3_1_8B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = False
    tie_embedding = False

    n_layers = 32
    vocab_size = 128256
    ffn_hidden_dim = 14336
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8

    rope_theta = 5e5
    rms_norm_eps = 1e-5
    attention_mask_value = -1e5

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = False
    fp16_qkv_heads = False
    n_fp16_heads = 0
    n_fp16_neurons = 0


class Llama3_2_1B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = False
    tie_embedding = True

    n_layers = 16
    vocab_size = 128256
    ffn_hidden_dim = 8192
    head_dim = 64
    n_heads = 32
    n_kv_heads = 8

    rope_theta = 5e5
    rms_norm_eps = 1e-5
    attention_mask_value = -1e5

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = False
    fp16_qkv_heads = False
    n_fp16_heads = 0
    n_fp16_neurons = 0


class Llama2_7B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = False
    tie_embedding = False

    n_layers = 32
    vocab_size = 32000
    ffn_hidden_dim = 11008
    head_dim = 128
    n_heads = 32
    n_kv_heads = 32

    rope_theta = 10000.0
    rms_norm_eps = 1e-5
    attention_mask_value = -1e6

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = False
    fp16_qkv_heads = False
    n_fp16_heads = 0
    n_fp16_neurons = 0


class Mistral_7B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = True
    tie_embedding = False

    n_layers = 32
    vocab_size = 32000
    ffn_hidden_dim = 14336
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8

    rope_theta = 1e4
    rms_norm_eps = 1e-5
    attention_mask_value = -1e2

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = False
    fp16_qkv_heads = False
    n_fp16_heads = 0
    n_fp16_neurons = 0


class Qwen2_5_7B_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = False

    n_layers = 28
    vocab_size = 152064
    ffn_hidden_dim = 18944
    head_dim = 128
    n_heads = 28
    n_kv_heads = 4

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = [0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27]
    fp16_ffn_layers = [0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27]
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = 0


class Qwen2_5_7B_Spinquant_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = False

    n_layers = 28
    vocab_size = 152064
    ffn_hidden_dim = 18944
    head_dim = 128
    n_heads = 28
    n_kv_heads = 4

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = 0


class Qwen2_5_500M_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 24
    vocab_size = 151936
    ffn_hidden_dim = 4864
    head_dim = 64
    n_heads = 14
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23]
    fp16_ffn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23]
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = 0


class Qwen2_5_500M_Spinquant_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 24
    vocab_size = 151936
    ffn_hidden_dim = 4864
    head_dim = 64
    n_heads = 14
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = 0


class SmallThinker_3B_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 36
    vocab_size = 151936
    ffn_hidden_dim = 11008
    head_dim = 128
    n_heads = 16
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = list(range(36))
    fp16_ffn_layers = []
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = {
        **{i: 1024 for i in range(36)},
        **{i: 11008 for i in [1]},
    }


class SmallThinker_500M_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 24
    vocab_size = 151936
    ffn_hidden_dim = 4864
    head_dim = 64
    n_heads = 14
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = [1, 9, 10, 11, 12, 13, 14, 17, 20, 21]
    fp16_ffn_layers = [14, 15, 16, 19]
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = 0


class SmallThinker_3B_Spinquant_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 36
    vocab_size = 151936
    ffn_hidden_dim = 11008
    head_dim = 128
    n_heads = 16
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = 0


class SmallThinker_500M_Spinquant_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 24
    vocab_size = 151936
    ffn_hidden_dim = 4864
    head_dim = 64
    n_heads = 14
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = []
    fp16_ffn_layers = []
    fp16_rope = True
    fp16_qkv_heads = True
    n_fp16_heads = 0
    n_fp16_neurons = 0


model_map: dict[str, ModelParams] = {
    "mistral_7b": Mistral_7B_Params,
    "qwen2_7b": Qwen2_5_7B_Params,
    "qwen2.5_7b": Qwen2_5_7B_Params,
    "qwen2.5_7b_spin": Qwen2_5_7B_Spinquant_Params,
    "qwen2.5_0.5b": Qwen2_5_500M_Params,
    "qwen2.5_0.5b_spin": Qwen2_5_500M_Spinquant_Params,
    "llama3_1_8b": Llama3_1_8B_Params,
    "llama3_2_1b": Llama3_2_1B_Params,
    "llama2_7b": Llama2_7B_Params,
    "smallthinker_3b": SmallThinker_3B_Params,
    "smallthinker_0.5b": SmallThinker_500M_Params,
    "smallthinker_3b_spin": SmallThinker_3B_Spinquant_Params,
    "smallthinker_0.5b_spin": SmallThinker_500M_Spinquant_Params,
}
