# 1. 概览

## 1.1 本节内容
量化部署

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
复制python环境：  
通过 `conda create --clone`
```bash
conda create -n lmdeploy --clone /share/conda_envs/internlm-base
```
或者
```bash
/root/share/install_conda_env_internlm_base.sh lmdeploy
```
activate
```bash
conda activate lmdeploy
```
安装lmdeploy包
```bash
pip install 'lmdeploy[all]==v0.1.0'
```

# 3. 模型转换