#!/bin/bash
# bash ./tools/gen_flame_graph.sh ./build/bin/run --file-path ../models/Meta-Llama-3.1-8B/llama3-8b_Q4_0.gguf --vocab-path ../models/Meta-Llama-3.1-8B/llama3.1_8b_vocab.gguf --prompt "Tell me a story:" --steps 16
set -x

TOOLS_DIR="/home/zwb/SS/FlameGraph"
CUR_DIR=$(pwd)

cmd=("$@")

sudo perf record -F 99 -a -g -- "${cmd[@]}"
sudo perf script -i perf.data > $CUR_DIR/out.perf
sudo $TOOLS_DIR/stackcollapse-perf.pl $CUR_DIR/out.perf > $CUR_DIR/out.folded
sudo $TOOLS_DIR/flamegraph.pl $CUR_DIR/out.folded > $CUR_DIR/out.svg
sudo chmod a+rw $CUR_DIR/out.svg

sudo rm -f $CUR_DIR/perf.data $CUR_DIR/out.perf $CUR_DIR/out.folded
