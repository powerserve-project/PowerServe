# PowerServe.Server

## Usage

Start a server:
- `--work-folder`(optional): The folder of model workspace. When receiving a request specifying the model, it search firstly in this directory. For example, when I specify `--work-folder model`, it will search models in the directory **./model** if it exists. When I specify the model folder with such a request:
    ```json
    {
        "model": "SmallThinker-3B-PowerServe-QNN29-8Gen4",
        "prompt": "Hello"
    }
    ```
    it will search the directory **./model/SmallThinker-3B-PowerServe-QNN29-8Gen4** at first and then **SmallThinker-3B-PowerServe-QNN29-8Gen4** if the former one does not exist.
- `--lib-folder`(optional): The folder containing QNN libraries. If not specified, it will be assigned with `model-folder/qnn_libs` by default.
- `--host`(optional): The IP address the server listen
- `--port`(optional): The IP port the server listen
```shell
./build/bin/server --work-folder model --host 127.0.0.1 --port 8080
```

Test the server simply:
- Completion
    ```shell
    curl --request POST \
        --url http://localhost:8080/completion \
        --header "Content-Type: application/json" \
        --data '{"prompt": "Once upon a time", "max_tokens": 128, "model": "SmallThinker-3B-PowerServe-QNN29-8Gen4"}'
    ```
- Chat
    ```shell
    curl --request POST \
        --url http://localhost:8080/v1/chat/completions \
        --header "Content-Type: application/json" \
        --data '{"messages": [{"role":"user", "content":"Once upon a time"}], "model": "SmallThinker-3B-PowerServe-QNN29-8Gen4"}'
    ```
- Streamly chat
    ```shell
    curl --request POST \
        --url http://localhost:8080/v1/chat/completions \
        --header "Content-Type: application/json" \
        --data '{"stream": true, "messages": [{"role":"user", "content":"Once upon a time"}], "model": "SmallThinker-3B-PowerServe-QNN29-8Gen4"}'
    ```
- Speculative chat: If you want server use speculative, please use '+' to connect 2 models.
    ```shell
    curl --request POST \
        --url http://localhost:8080/v1/chat/completions \
        --header "Content-Type: application/json" \
        --data '{"stream": true, "messages": [{"role":"user", "content":"Once upon a time"}], "model": "SmallThinker-3B-PowerServe-QNN29-8Gen4+SmallThinker-0.5B-PowerServe-QNN29-8Gen4"}'
    ```

## OpenAI API

PowerServe.server support part of OpenAI API:
- `/v1/completions`
- `/v1/chat/completions`
- `/v1/models`

As for the detail please refer to the [API documentation](https://platform.openai.com/docs/api-reference)
