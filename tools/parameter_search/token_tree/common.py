import itertools
import json
import os
import random
from pathlib import Path


# search_grid = {
#     "draft_top_k": [3, 5, 10, 15, 20, 25],
#     "draft_temperature": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
#     "draft_p_base": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     "tree_max_fan_out": [1, 2, 3, 4, 5, 6],
#     "tree_min_prob": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7],
#     "tree_early_stop": [0, 1],
# }

search_grid = {
    "draft_top_k": [10, 15, 20],
    "draft_temperature": [1.2, 1.4, 1.5, 1.6, 1.7, 1.8],
    "draft_p_base": [0.6, 0.75, 0.9],
    "tree_max_fan_out": [3, 5, 6, 7, 8, 9],
    "tree_min_prob": [0.1, 0.15, 0.175, 0.2, 0.25],
    "tree_early_stop": [1],
}


keys = list(search_grid.keys())
value_lists = list(search_grid.values())
database = {tuple(zip(keys, values)): None for values in itertools.product(*value_lists)}

print(f"#combination={len(database)}")

database_path = Path("./database.jsonl")

if database_path.exists():
    n_loaded = 0
    with open(database_path, "r") as database_file:
        for line in database_file:
            line = line.strip()
            if len(line) == 0:
                continue

            data = json.loads(line)
            params = tuple(data["params"].items())
            stat = data["stat"]

            if params in database:
                assert database[params] is None
                database[params] = stat
                n_loaded += 1

    print(f"Loaded {n_loaded} database entries.")

database_file = open(database_path, "a")

untested_params = [params for params, stat in database.items() if stat is None]
print(f"{len(untested_params)} untested params.")

random.shuffle(untested_params)


def format_params(params: tuple) -> str:
    return " ".join(f"{key}={value}" for key, value in params)


def run(params: tuple) -> dict:
    assert params in database
    assert database[params] is None

    dump_file_path = Path("./stat.json")
    dump_file_path.unlink(missing_ok=True)

    command = " ".join([
        f"dump_file={dump_file_path}",
        format_params(params),
        "LD_LIBRARY_PATH=/system/lib64:/vendor/lib64",
        "sudo -E ./powerserve-speculative --work-folder smallthinker --prompt-file comparison_qwen2.txt -n 1024",
    ])

    print(f"> {command}")
    assert os.system(command) == 0
    assert dump_file_path.exists()

    with open(dump_file_path, "r") as dump_file:
        stat = json.load(dump_file)

    database[params] = stat

    database_file.write(
        json.dumps({
            "params": dict(params),
            "stat": stat,
        })
        + "\n"
    )
    database_file.flush()
