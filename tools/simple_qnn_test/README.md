# Simple QNN Test

## Convert Model

convert model + compile model + generate binary
```shell
python3 simple_convert_qnn.py --model-name simple_model --model-num 3
```
The output will locate in the directory `output`

## Execute Model

Push the model to the phone

```shell
adb push output/simple_model.bin
```

```shell
<path-to-build-dir>/tests/qnn_tests --qnn-path /data/local/tmp/simple_model \
                --model-name <model-name>                                   \
                --graph-num <graph-num>                                     \
                --repeat <repeat-num>
```
