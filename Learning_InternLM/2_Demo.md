# 1. 内容简介

主要是InternLM中三个主要项目的Demo
+ InternLM/InternLM-Chat: 智能对话
+ InternLM/Lagent: 智能体
+ InternLM/XComposer: 多模态

# 2. Demo `InternLM-Chat` 
## 2.1 环境配置
```bash
bash

# 1. 克隆环境
/root/share/install_conda_env_internlm_base.sh internlm-demo

# 2. 激活环境
conda activate internlm-demo

# 3. 升级pip并安装依赖
python -m pip install --upgrade pip
pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```
## 2.2 模型文件
将模型文件复制过来
```bash
mkdir -p /root/model/Shanghai_AI_Laboratory
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/model/Shanghai_AI_Laboratory
```
或者下载模型文件
```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/root/model', revision='v1.0.3')
```
## 2.3 代码
### 2.3.1 git clone
```bash
cd /root/code
git clone https://gitee.com/internlm/InternLM.git
cd InternLM
git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17
```
### 2.3.2 修改`web_demo.py`  
将`/root/code/InternLM/web_demo.py` 中29行和33行的模型地址更换为`/root/model/Shanghai_AI_Laboratory/internlm-chat-7b`

### 2.3.2 创建`cli_demo.py`
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"

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
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```
## 2.3 运行
### 2.3.1 cli_demo
```bash
python /root/code/InternLM/cli_demo.py
```
### 2.3.2 web_demo
```bash
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```


# 3. Demo `Lagent`

# 4. Demo `XComposer` 

# 5. 环境相关

## 5.1 InternStudio
[InterStudio官网](https://studio.intern-ai.org.cn/console/dashboard)
### 5.1.1 创建开发机
1. 设置"开发机名称"
2. 选择并使用cuda镜像
3. 选择"资源配置"
4. 设置"运行时长"
5. 点击"立即创建"
### 5.1.2 开发机使用
+ JupyterLab
+ Terminal
+ VSCode

注意:InternLM中，在首次使用Terminal时请进入`bash`环境

## 5.2 SSH
### 5.2.1 SSH密钥配置
1. 在本地通过`ssh-keygen -t rsa`命令生成密钥对，指定文件位置`C:\Users\dukai\.ssh\InternStudio`
2. 查看生成的公钥`cat ~\.ssh\InternStudio.pub`，并将其添加到InternStudio的SSH配置中
### 5.2.2 SSH端口转发
```bash
ssh -CNg -L 6006:127.0.0.1:6006 -i  C:/Users/dukai/.ssh/InternStudio root@ssh.intern-ai.org.cn -p 33090
``` 
命令解读：在本地监听`6006`端口，并将流量转发到远程主机`ssh.intern-ai.org.cn`的`33090`端口(`-i`用来指定私钥文件),再由远程主机转发到`127.0.0.1:6006`