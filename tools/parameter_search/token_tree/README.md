首先需要导出模型和编译项目。

超参数设置修改`common.py`中的`search_grid`，运行命令修改`run`函数。

推送脚本到手机上：

```bash
rsync -avzP tools/parameter_search/token_tree/{common,search}.py 8gen4:~/
```

在手机上运行脚本：

```bash
python search.py
```

然后将数据库下载到本地，并用`analyze.py`分析：

```bash
cd tools/parameter_search/token_tree
rsync -avzP 8gen4:~/database.jsonl .
python analyze.py
```
