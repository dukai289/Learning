# 1. 概览

## 1.1 本节内容
使用轻量级微调框架 `InternLM/Xtuner` 实现三个功能：
+ 大规模对齐：在数据集 `OASST1` 上微调
+ 医学问答能力：在数据集 `MedQA` 上微调
+ Agent能力：在数据集 `MS-Agent` 上微调

## 1.2 微调的准备
+ LLM
+ Dataset
+ 微调脚本

## 1.3 微调过程
1. 处理数据集
2. 配置微调脚本
2. 利用Xtuner训练得到PTH文件
3. 将PTH文件转化成hf-Adapter文件
4. 将LLM与hf-Adapter文件进行Merge
5. 部署与测试

# 2. 环境准备
进入 `bash`
```bash
bash
```
python环境
```bash
/root/share/install_conda_env_internlm_base.sh xtuner0.1.9
conda activate xtuner0.1.9
```
xtuner环境
```python
cd ~
mkdir xtuner019 && cd xtuner019

git clone -b v0.1.9  https://github.com/InternLM/xtuner
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner 无法访问github时请从gitee拉取:
cd xtuner
pip install -e '.[all]'

export PATH=$PATH:~/.local/bin

xtuner list-cfg
```

# 3. 大规模对齐
通过在 `OASST1` 上微调，使得 `InternLm-7B` 实现大规模对齐
## 3.1 工作目录
```bash
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
```

## 3.2 LLM
从开发机的 `model_repos` 目录复制
```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b ~/ft-oasst1/
```
或者，从 `modelscope` 下载
```bash
mkdir ~/ft-oasst1/internlm-chat-7b

pip install modelscope

cd ~/ft-oasst1
apt install git git-lfs -y
git lfs install
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm-chat-7b.git -b v1.0.3
```

## 3.3 数据集
```bash
cd ~/ft-oasst1
cp -r /root/share/temp/datasets/openassistant-guanaco .
```

## 3.4 xtuner配置文件
复制一份配置文件到当前目录
```bash
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```
修改配置文件
```bash
vim internlm_chat_7b_qlora_oasst1_e3_copy.py
```
```diff
# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据集为本地路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = './openassistant-guanaco'
```



## 3.5 微调
### 3.5.1 目录结构
查看 `~/ft-oasst1/` 目录结构
```bash
# apt-get install tree
tree ~/ft-oasst1/
```
结果应该如下
```bash
|-- internlm-chat-7b
|   |-- README.md
|   |-- config.json
|   |-- configuration.json
|   |-- configuration_internlm.py
|   |-- generation_config.json
|   |-- modeling_internlm.py
|   |-- pytorch_model-00001-of-00008.bin
|   |-- pytorch_model-00002-of-00008.bin
|   |-- pytorch_model-00003-of-00008.bin
|   |-- pytorch_model-00004-of-00008.bin
|   |-- pytorch_model-00005-of-00008.bin
|   |-- pytorch_model-00006-of-00008.bin
|   |-- pytorch_model-00007-of-00008.bin
|   |-- pytorch_model-00008-of-00008.bin
|   |-- pytorch_model.bin.index.json
|   |-- special_tokens_map.json
|   |-- tokenization_internlm.py
|   |-- tokenizer.model
|   `-- tokenizer_config.json
|-- internlm_chat_7b_qlora_oasst1_e3_copy.py
`-- openassistant-guanaco
    |-- openassistant_best_replies_eval.jsonl
    `-- openassistant_best_replies_train.jsonl
```
### 3.5.2 进行微调
```bash
# NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero2
nohup xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2 >> ./log_xtuner.txt 2>&1 &
```  
### 3.5.3 日志
微调时会生成 `work_dirs` 目录
```bash
.
|-- internlm-chat-7b
|   |-- README.md
|   |-- config.json
|   |-- configuration.json
|   |-- configuration_internlm.py
|   |-- generation_config.json
|   |-- modeling_internlm.py
|   |-- pytorch_model-00001-of-00008.bin
|   |-- pytorch_model-00002-of-00008.bin
|   |-- pytorch_model-00003-of-00008.bin
|   |-- pytorch_model-00004-of-00008.bin
|   |-- pytorch_model-00005-of-00008.bin
|   |-- pytorch_model-00006-of-00008.bin
|   |-- pytorch_model-00007-of-00008.bin
|   |-- pytorch_model-00008-of-00008.bin
|   |-- pytorch_model.bin.index.json
|   |-- special_tokens_map.json
|   |-- tokenization_internlm.py
|   |-- tokenizer.model
|   `-- tokenizer_config.json
|-- internlm_chat_7b_qlora_oasst1_e3_copy.py
|-- openassistant-guanaco
|   |-- openassistant_best_replies_eval.jsonl
|   `-- openassistant_best_replies_train.jsonl
`-- work_dirs
    `-- internlm_chat_7b_qlora_oasst1_e3_copy
        |-- 20240110_145558
        |   |-- 20240110_145558.log
        |   `-- vis_data
        |       |-- 20240110_145558.json
        |       |-- config.py
        |       `-- scalars.json
        `-- internlm_chat_7b_qlora_oasst1_e3_copy.py
```
其中日志文件在 `work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy` 下，可以用 `tail -f -n 30 {log.txt}` 命令来查看。  
### 3.5.4 微调完成
微调完成后，再次查看 `~/ft-oasst1/` 目录结构，结果应当如下
```bash

```
### 3.5.5 转换
将得到的 PTH 模型转换为 HuggingFace 模型，即：生成 Adapter 文件夹
```bash
# xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```
再次查看 `~/ft-oasst1/` 目录结构，结果应当如下
```bash

```
### 3.5.6 合并
将 HuggingFace adapter 合并到大语言模型
```bash
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

## 3.6 测试与部署
进行一下对话测试
```bash
# 加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm_chat

# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```
修改 `cli_demo.py` 文件
```diff
- model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"
+ model_name_or_path = "merged"
```
运行 `cli_demo`
```bash
python ./cli_demo.py
```


# 4. 医学知识
通过在 `MedQA` 上微调，使得 `InternLm-7B` 获得医学知识
## 4.1 工作目录
```bash
mkdir ~/ft-medqa && cd ~/ft-medqa
```
## 4.2 LLM
```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b .
```

# 5. Agent能力
通过在 `MS-Agent` 上微调，使得 `InternLm-7B` 获得Agent能力
## 5.1 工作目录
```bash
mkdir ~/ft-msagent && cd ~/ft-msagent
```
## 5.2 LLM
```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b .
```