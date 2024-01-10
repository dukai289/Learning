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
python环境
```bash
bash
/root/share/install_conda_env_internlm_base.sh xtuner0.1.9
conda activate xtuner0.1.9

cd ~
mkdir xtuner019 && cd xtuner019
```
xtuner环境
```python
git clone -b v0.1.9  https://github.com/InternLM/xtuner
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner 无法访问github时请从gitee拉取:

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```

# 3 大规模对齐
通过在 `OASST1` 上微调，使得 `InternLm-7B` 实现大规模对齐
## 3.1 工作目录
```bash
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
```
## 3.2 LLM
```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b ~/ft-oasst1/
```

# 4 医学知识
通过在 `MedQA` 上微调，使得 `InternLm-7B` 获得医学知识
## 4.1 工作目录
```bash
mkdir ~/ft-medqa && cd ~/ft-medqa
```
## 4.2 LLM
```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b .
```

# 5 Agent能力
通过在 `MS-Agent` 上微调，使得 `InternLm-7B` 获得Agent能力
## 5.1 工作目录
```bash
mkdir ~/ft-msagent && cd ~/ft-msagent
```
## 5.2 LLM
```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b .
```