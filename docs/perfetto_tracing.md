首先编译代码时要设置`POWERSERVE_WITH_PERFETTO`为`ON`。

然后要在代码里显示地启用和停止tracing，例如：

```c++
powerserve::PerfettoTrace::instance().start_tracing(32 * 1024); // 这里填的是缓冲区大小，单位KiB，需要大于最终导出的trace文件大小
powerserve::TreeSpeculative spec(main_model, draft_model);
spec.generate(tokenizer, sampler, prompt, n_predicts);
spec.print_stat();
powerserve::PerfettoTrace::instance().stop_tracing("./perfetto.data"); // 最后会保存到perfetto.data文件
```

用`PerfettoTrace::begin`和`PerfettoTrace::end`标记开始和结束的位置，例如：

```c++
PerfettoTrace::begin("draft_model_forward");
auto logits = draft_model->forward({node.token}, {node.position}, CausalAttentionMask(1));
PerfettoTrace::end();
```

`PerfettoTrace::counter`可以用来记录一条折线图。

最后，在<https://ui.perfetto.dev>打开trace文件。
