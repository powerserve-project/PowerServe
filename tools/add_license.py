import argparse
import os
from pathlib import Path
from typing import List


license = """// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

"""

license_py = """# Copyright 2024-2025 PowerServe Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

root_folder = Path(".").absolute()


def check_and_add_header(file_path: Path, header_content: str, quiet: bool):
    if not quiet:
        print(file_path)

    with open(file_path, "r+", encoding="utf-8") as f:
        content = f.read()
        # 检查文件头部是否包含指定内容
        if header_content not in content:
            # 添加内容到文件头部
            f.seek(0, 0)
            f.write(header_content + "\n" + content)

            if not quiet:
                print(f"Added header to {file_path}")


def find_files(directory: Path, quiet: bool = False):
    targets: List[Path] = []

    if not directory.is_dir():
        target = Path(os.path.join(root_folder, directory))
        targets.append(target)
    else:
        for root, dirs, files in os.walk(directory):
            for file in files:
                target = Path(os.path.join(root, file))
                if target.suffix in [".cpp", ".hpp", ".c", ".h"]:
                    targets.append(target)

    for target in targets:
        if target.suffix in [".cpp", ".hpp", ".c", ".h"]:
            check_and_add_header(target, license, quiet)
        elif target.suffix in [".py"]:
            check_and_add_header(target, license_py, quiet)


def main():
    parser = argparse.ArgumentParser(prog="PowerServe", description="PowerServe License Add Tool")
    parser.add_argument("-d", "--dir", type=Path, required=True, help="file path or dir path")
    args = parser.parse_args()

    find_files(args.dir)


if __name__ == "__main__":
    main()
