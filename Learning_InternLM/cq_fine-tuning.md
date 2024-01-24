# 1. 主题
 - 微调
 - RAG

# 2. 大纲
## 2.1 要素
 - 训练集
 - LLM
 - 微调框架
 - 开发机
## 2.2 微调过程
 1. 环境准备
    1. 工作目录
    2. conda环境
    3. fine-tuning框架
 2. 将原始数据处理为训练集
    1. 上传原始数据
    2. 编写处理脚本
    3. 生成训练集
    4. 存储训练集
 3. 准备LLM
 4. 配置微调脚本
 5. 执行微调
 6. 微调后处理
    1. convert
    2. merge
    3. 测试
    4. 存储
 7. 部署
## 2.3 难点
 - 训练集构造：从文本到问答形式
 - 训练参数的选择


# 3. 环境准备
工作目录 `~/ft-cq`
```bash
bash

mkdir ~/ft-cq &&　cd  ~/ft-cq
```
conda环境
```bash
/root/share/install_conda_env_internlm_base.sh ft-cq
conda activate ft-cq
```
fine-tuning框架
```bash
cd ~
mkdir xtuner019 && cd xtuner019

git clone -b v0.1.9  https://github.com/InternLM/xtuner
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner 无法访问github时请从gitee拉取:
cd xtuner
pip install -e '.[all]'

export PATH=$PATH:~/.local/bin

xtuner list-cfg
```
返回工作目录 `~/ft-cq`
```bash
cd ~/ft-cq
```
# 4. 训练集
 -> `dataset_cq.jsonl`
## 4.1 上传原始数据
## 4.2 数据转换脚本
`data/xlsx2jsonl.py`
```python
import pandas as pd
import json

file = r'./data/知识库内容.xlsx'
df = pd.read_excel(file)
print(df.head())

output_data = []
for idx,row in df.iterrows():
    system_value = "You are a professional, highly experienced doctor professor. You always provide accurate, comprehensive, and detailed answers based on the patients' questions."
    input_row = '关于`{}`的`{}`问题,{}?'.format(row['info_1'], row['info_2'], row['Q'])
    conversation = {
        "system": system_value,
        "input": input_row,
        "output": row['A']
    }
    # print(conversation)
    output_data.append({"conversation": [conversation]})
print(len(output_data))

output_file =r'./data/dataset_cq.jsonl'
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, indent=4,ensure_ascii=False)

```

## 4.3 处理数据
```bash
python data/xlsx2jsonl.py
```
## 4.4 查看数据

# 5. LLM
```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b ~/ft-cq
```
# 6. 微调脚本
复制配置文件
```bash
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_cq_e3.py
```
修改配置文件内容
```bash
vim internlm_chat_7b_qlora_cq_e3.py
```

```diff
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据为 MedQA2019-structured-train.jsonl 路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = './data/dataset_cq.jsonl'

# 修改 train_dataset 对象
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
```


# 7. 执行微调
`tree .` 查看当前目录结构
```bash
.
|-- data
|   |-- dataset_cq.jsonl
|   |-- xlsx2jsonl.py
|   `-- \347\237\245\350\257\206\345\272\223\345\206\205\345\256\271.xlsx
|-- dataset_cq.jsonl
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
`-- internlm_chat_7b_qlora_cq_e3.py
```
执行微调
```bash
nohup xtuner train internlm_chat_7b_qlora_cq_e3.py --deepspeed deepspeed_zero2 >> ./log_xtuner.txt 2>&1 &
```
查看日志
```bash
tail -f -n 30 log_xtuner.txt
```

# 8. 微调后处理
## 8.1 convert
将得到的 PTH 模型转换为 HuggingFace 模型，即：生成 Adapter 文件夹
```bash
# xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf ./internlm_chat_7b_qlora_cq_e3.py ./work_dirs/internlm_chat_7b_qlora_cq_e3/epoch_1.pth ./hf
```
再次查看 `~/ft-cq/` 目录结构，结果应当如下
```bash
.
|-- data
|   |-- dataset_cq.jsonl
|   |-- xlsx2jsonl.py
|   `-- \347\237\245\350\257\206\345\272\223\345\206\205\345\256\271.xlsx
|-- dataset_cq.jsonl
|-- hf
|   |-- README.md
|   |-- adapter_config.json
|   |-- adapter_model.safetensors
|   `-- xtuner_config.py
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
|-- internlm_chat_7b_qlora_cq_e3.py
|-- log_xtuner.txt
`-- work_dirs
    `-- internlm_chat_7b_qlora_cq_e3
        |-- 20240117_120122
        |   |-- 20240117_120122.log
        |   `-- vis_data
        |       |-- 20240117_120122.json
        |       |-- config.py
        |       `-- scalars.json
        |-- epoch_1.pth
        |   |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |   `-- mp_rank_00_model_states.pt
        |-- epoch_2.pth
        |   |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |   `-- mp_rank_00_model_states.pt
        |-- epoch_3.pth
        |   |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |   `-- mp_rank_00_model_states.pt
        |-- internlm_chat_7b_qlora_cq_e3.py
        |-- last_checkpoint
        `-- zero_to_fp32.py
```
## 8.2 merge
将 HuggingFace adapter 合并到大语言模型
```bash
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```
再次查看 `~/ft-cq/` 目录结构，结果应当如下
```bash
.
|-- data
|   |-- dataset_cq.jsonl
|   |-- xlsx2jsonl.py
|   `-- \347\237\245\350\257\206\345\272\223\345\206\205\345\256\271.xlsx
|-- dataset_cq.jsonl
|-- hf
|   |-- README.md
|   |-- adapter_config.json
|   |-- adapter_model.safetensors
|   `-- xtuner_config.py
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
|-- internlm_chat_7b_qlora_cq_e3.py
|-- log_xtuner.txt
|-- merged
|   |-- added_tokens.json
|   |-- config.json
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
`-- work_dirs
    `-- internlm_chat_7b_qlora_cq_e3
        |-- 20240117_120122
        |   |-- 20240117_120122.log
        |   `-- vis_data
        |       |-- 20240117_120122.json
        |       |-- config.py
        |       `-- scalars.json
        |-- epoch_1.pth
        |   |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |   `-- mp_rank_00_model_states.pt
        |-- epoch_2.pth
        |   |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |   `-- mp_rank_00_model_states.pt
        |-- epoch_3.pth
        |   |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |   `-- mp_rank_00_model_states.pt
        |-- internlm_chat_7b_qlora_cq_e3.py
        |-- last_checkpoint
        `-- zero_to_fp32.py
```
## 8.3 测试
```bash
vim /root/xtuner019/xtuner/xtuner/tools/chat.py
```
修改

进行一下对话测试
```bash
# 加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm_chat

# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```
## 8.4 上传
将微调好的模型上传到modelscope
```

```

# 9. 部署
新建 `cli_demo.py` 文件
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "./merged"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```
运行 `cli_demo`
```bash
python ./cli_demo.py
```