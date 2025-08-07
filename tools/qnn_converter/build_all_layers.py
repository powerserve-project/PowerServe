import argparse
import multiprocessing
import subprocess
from pathlib import Path

from soc_config import soc_map


parser = argparse.ArgumentParser()
parser.add_argument("--n-threads", type=int, required=True)
parser.add_argument("--build-folder", type=Path, required=True)
parser.add_argument("--batch-size", default=-1, type=int)
parser.add_argument("--n-model-chunks", type=int, required=True)
parser.add_argument("--artifact-name", type=str, required=True)
parser.add_argument("--graph-names", type=str, nargs="+", required=True)
parser.add_argument("--soc", choices=soc_map.keys(), default="8650")
parser.add_argument("--silent", action="store_true", help="Hide the shell command arguments.")
args = parser.parse_args()


def run(cmd_args: list):
    cmd = " ".join(map(str, cmd_args))
    if args.silent:
        cmd_short = " ".join(map(str, cmd_args[:2]))
        print(f"> {cmd_short}")
    else:
        print(f"> {cmd}")
    ret = subprocess.Popen(cmd, shell=True).wait()
    assert ret == 0


root_folder = Path(__file__).parent.absolute()


def build_shared_object(chunk_id: int):
    batch_size = args.batch_size
    build_folder = args.build_folder.absolute()

    # os.chdir(build_folder)

    if chunk_id == -1:  # -1: Perform the output embedding conversion
        subdir = "output_embedding"
    else:
        subdir = f"model_chunk_{chunk_id}"

    command = [
        "python",
        root_folder / "build_shared_object.py",
        "--model",
        build_folder / subdir / f"batch_{batch_size}" / "onnx_model" / f"batch_{batch_size}.onnx",
        "--encoding",
        build_folder / subdir / f"batch_{batch_size}" / f"batch_{batch_size}.encodings",
        "--io-spec",
        build_folder / subdir / f"batch_{batch_size}" / f"batch_{batch_size}.io.json",
        "--input-list",
        build_folder / subdir / f"batch_{batch_size}" / "input_list.txt",
        "--output-folder",
        build_folder / subdir,
        "--artifact-name",
        f"{args.artifact_name}_{chunk_id}" if chunk_id == -1 else "lm_head",
        "--graph-names",
        " ".join(args.graph_names),
    ]

    if args.silent:
        command.insert(2, "--silent")
    run(command)


def build_binary(chunk_id: int):
    build_folder = args.build_folder.absolute()

    # os.chdir(build_folder)

    if chunk_id == -1:  # -1: Perform the output embedding conversion
        subdir = "output_embedding"
    else:
        subdir = f"model_chunk_{chunk_id}"
    command = [
        "python",
        root_folder / "generate_binary.py",
        "--build-folder",
        build_folder / subdir,
        "--artifact-name",
        f"{args.artifact_name}_{chunk_id}" if chunk_id != -1 else "lm_head",
        "--graph-names",
        " ".join(args.graph_names),
        "--soc",
        args.soc,
    ]

    if args.silent:
        command.insert(2, "--silent")
    run(command)


multiprocessing.Process()
pool = multiprocessing.Pool(args.n_model_chunks if args.n_model_chunks < args.n_threads else args.n_threads)
chunk_ids = list(range(args.n_model_chunks))
chunk_ids.insert(0, -1)
if args.batch_size == -1:
    pool.map(build_binary, chunk_ids)
else:
    pool.map(build_shared_object, chunk_ids)
