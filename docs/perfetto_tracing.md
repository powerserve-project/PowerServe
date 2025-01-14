First, set POWERSERVE_WITH_PERFETTO to ON during code compilation.

Then explicitly start and stop tracing in your code, for example:

```c++
powerserve::PerfettoTrace::instance().start_tracing(32 * 1024); // Buffer size in KiB 
powerserve::TreeSpeculative spec(main_model, draft_model);
spec.generate(tokenizer, sampler, prompt, n_predicts);
spec.print_stat();
powerserve::PerfettoTrace::instance().stop_tracing("./perfetto.data"); // Will save to perfetto.data file
```

Use PerfettoTrace::begin and PerfettoTrace::end to mark start and end positions, for example:

```c++
PerfettoTrace::begin("draft_model_forward");
auto logits = draft_model->forward({node.token}, {node.position}, CausalAttentionMask(1));
PerfettoTrace::end();
```

`PerfettoTrace::counter`can be used to record a line graph.

Finally, open the trace file at https://ui.perfetto.dev.
