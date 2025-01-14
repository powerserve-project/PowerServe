class GraphParams:
    batch_size: int
    cache_size: int
    context_size: int


class Batch1_Params(GraphParams):
    batch_size = 1
    cache_size = 1920
    context_size = 2048


class Batch4_Params(GraphParams):
    batch_size = 4
    cache_size = 1920
    context_size = 2048


class Batch8_Params(GraphParams):
    batch_size = 8
    cache_size = 1920
    context_size = 2048


class Batch12_Params(GraphParams):
    batch_size = 12
    cache_size = 1920
    context_size = 2048


class Batch16_Params(GraphParams):
    batch_size = 16
    cache_size = 1920
    context_size = 2048


class Batch32_Params(GraphParams):
    batch_size = 32
    cache_size = 1920
    context_size = 2048


class Batch128_Params(GraphParams):
    batch_size = 128
    cache_size = 1920
    context_size = 2048


graph_map: dict[str, GraphParams] = {
    "batch_1": Batch1_Params,
    "batch_4": Batch4_Params,
    "batch_8": Batch8_Params,
    "batch_12": Batch12_Params,
    "batch_16": Batch16_Params,
    "batch_32": Batch32_Params,
    "batch_128": Batch128_Params,
}
