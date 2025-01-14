# MMLU Test
- Download the test set: https://github.com/bobozi-cmd/mmlu/releases/download/publish/mmlu_dataset.zip

- Unpack and put it in the same directory as the script `mmlu_test.py`:
```
> tree -L 1 ./
./
├── dev
├── mmlu_test.py
├── README.md
├── test
└── val
```

- Start the server, assuming that the IP address is 192.168.1.39, the model directory is build_llama_proj, and the port is 8080. Test command:

```bash
python ./mmlu_test.py --host 192.168.1.39 --port 8080 -s 1 --model build_llama_proj
```

- MMLU has a total of 57 subjects, and `-s30` means the first 30 questions in each subject. Without adding the '-s' parameter, you will measure the entire MMLU data set.
- If you run on pure CPU, it will be slower, so you can test `-s1` first.
