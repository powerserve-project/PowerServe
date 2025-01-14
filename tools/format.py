# Format all files:
# git ls-files | xargs python tools/format.py --no-diff -j4

import argparse
import os
from pathlib import Path

from tqdm.contrib.concurrent import process_map


parser = argparse.ArgumentParser()
parser.add_argument("--no-diff", action="store_true", help="Do not diff after formatting")
parser.add_argument("-j", "--num-threads", type=int, default=1)
parser.add_argument("files", type=Path, nargs="+")
args = parser.parse_args()


def is_empty(text: str) -> bool:
    return text.strip() == ""


def read(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


def write(path: Path, text: str):
    with open(path, "w") as f:
        f.write(text)


def remove_head_empty_lines(text: str) -> str:
    lines = text.split("\n")
    while len(lines) >= 1 and is_empty(lines[0]):
        lines = lines[1:]
    return "\n".join(lines)


def ensure_tail_empty_line(text: str) -> str:
    lines = text.split("\n")
    while len(lines) >= 2 and is_empty(lines[-1]) and is_empty(lines[-2]):
        lines = lines[:-1]
    if len(lines) == 0 or not is_empty(lines[-1]):
        lines.append("")
    return "\n".join(lines)


def remove_trailing_spaces(text: str) -> str:
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    return "\n".join(lines)


def compact_consecutive_empty_lines(text: str) -> str:
    lines = text.split("\n")
    new_lines = []
    for i in range(len(lines)):
        if i + 3 <= len(lines) and is_empty(lines[i]) and is_empty(lines[i + 1]) and is_empty(lines[i + 2]):
            continue
        else:
            new_lines.append(lines[i])
    return "\n".join(new_lines)


PROTECT_COMMENT = "//<Protected by format.py!!!>"


def protect_pragma(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#pragma"):
            lines[i] = line.replace("#pragma", PROTECT_COMMENT + "#pragma")
    return "\n".join(lines)


def clang_format(path: Path):
    clang_format_path = os.getenv("CLANG_FORMAT", "clang-format")
    assert os.system(f"{clang_format_path} -i {path}") == 0


def black_format(path: Path):
    assert os.system(f"black -q {path} -l 120") == 0


def isort_format(path: Path):
    assert os.system(f"isort -q --lines-after-imports=2 {path}") == 0


def add_license(path: Path):
    from add_license import find_files

    find_files(path, quiet=True)


def unprotect_pragma(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith(PROTECT_COMMENT + "#pragma"):
            lines[i] = line.replace(PROTECT_COMMENT + "#pragma", "#pragma")
    return "\n".join(lines)


def handle_normal_text_files(path: Path):
    text = read(path)
    text = remove_head_empty_lines(text)
    text = remove_head_empty_lines(text)
    text = compact_consecutive_empty_lines(text)
    text = ensure_tail_empty_line(text)
    text = remove_trailing_spaces(text)
    write(path, text)


def handle_cpp_sources(path: Path):
    text = add_license(path)
    text = read(path)
    text = remove_head_empty_lines(text)
    text = remove_head_empty_lines(text)
    text = ensure_tail_empty_line(text)
    text = remove_trailing_spaces(text)
    text = protect_pragma(text)
    write(path, text)
    clang_format(path)
    text = read(path)
    text = unprotect_pragma(text)
    write(path, text)


def handle_py_sources(path: Path):
    text = read(path)
    text = remove_head_empty_lines(text)
    text = remove_head_empty_lines(text)
    text = ensure_tail_empty_line(text)
    text = remove_trailing_spaces(text)
    write(path, text)
    black_format(path)
    isort_format(path)
    text = read(path)
    write(path, text)


def process_path(path: Path):
    if not path.exists():
        print(f"{path} does not exist")
        return

    if path.suffix in [".md", ".sh", ".yml", ".toml"] or path.name in [
        ".clang-format",
        ".gitignore",
        ".gitmodules",
        "CMakeLists.txt",
        "requirements.txt",
    ]:
        handle_normal_text_files(path)
    elif path.suffix in [".py"]:
        handle_py_sources(path)
    elif path.suffix in [".c", ".h", ".cpp", ".hpp"]:
        handle_cpp_sources(path)


process_map(process_path, args.files, max_workers=args.num_threads)

if not args.no_diff:
    ret = os.system("git -c color.ui=always diff --exit-code --ignore-submodules=dirty")
    print(f"git diff returns {ret}")
    exit(0 if ret == 0 else 1)
