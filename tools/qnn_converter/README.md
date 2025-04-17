# Convert the model in safetensors format to a QNN executable binary format
1. Set up the QNN environment
    - Complete the configuration of the QNN environment following the instructions at https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html?product=1601111740009302.
    - After completing the configuration, navigate to the current directory and set the environment variables as follows:
        ```sh
        export QNN_SDK=/path/to/aforementioned/QNN/installation/directory
        source $QNN_SDK/bin/envsetup.sh
        #If successful, it will display:
        #[INFO] AISW SDK environment set
        #[INFO] QNN_SDK_ROOT: /path/to/aforementioned/QNN/installation/directory
        #[INFO] SNPE_ROOT: /path/to/aforementioned/QNN/installation/directory
        ```
    - You should use the Python virtual environment that was configured when setting up the QNN environment.
        ```sh
        export PYTHON_VENV_PATH=/path/to/python/virtual/environment
        source $PYTHON_VENV_PATH/bin/activate
        pip install -r requirements.txt
        ```
2. Run the one-click conversion script to complete the conversion
    ```sh
    python converter.py \
    --model-folder Llama-3.2-1B-Instruct \
    --model-name llama3_2_1b \
    --system-prompt-file ./prompt/system_prompt_llama.txt \
    --prompt-file ./prompt/lab_intro_llama.md \
    --batch-sizes 1 128 \
    --artifact-name llama3_2_1b \
    --n-model-chunk 1 \
    --silent \
    --clear-build-files \
    --soc 8650
    # Note:
    # 1. System-prompt-file and lab_intro_llama.md should be adjusted based on whether it is an Instruct model and the model template
    # 2. The number of batch sizes supports up to two.
    # 3. The value of n-model-chunk should ensure that the size of each chunk does not exceed 2G and can evenly divide the number of model layers. For an 8B model, it is recommended to set it to 4.
    ```
    After the conversion is completed, copy the resulting output folder to the correct location  for the QNN model on the phone to run

# Model Statistics

You need to profile first, and save statistics as JSON files into a folder named `stat`, under the model folder.

- For attention: A list with attention head ids.
- For feed forward: A list with neuron ids.

These head/neuron ids should be sorted by importance (e.g. by quantization error) in descending order.

Example:

```bash
> ls -v1 /ssd/smallthinker_3b_20241220/stat
attn_0_stat.json
attn_1_stat.json
attn_2_stat.json
...
attn_35_stat.json
ffn_0_stat.json
ffn_1_stat.json
ffn_2_stat.json
...
ffn_35_stat.json
> cat /ssd/smallthinker_3b_20241220/stat/attn_0_stat.json
[13, 12, 10, 11, 9, 8, 14, 15, 2, 5, 1, 7, 4, 0, 3, 6]
```
