# 1. 内容简介

主要是InternLM中三个主要项目的Demo
+ InternLM/InternLM-Chat: 智能对话
+ InternLM/Lagent: 智能体
+ InternLM/XComposer: 多模态

# 2. InternLM-Chat Demo

# 3. Lagent Demo

# 4. XComposer Demo

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

注意:InternLM中，在首次使用Terminal时请进入*bash*环境
## 5.2 SSH
### 5.2.1 SSH密钥配置
1. 在本地通过*ssh-keygen -t rsa*命令生成密钥对，指定文件位置* C:\Users\dukai/.ssh/InternStudio*
2. 将生成的公钥(cat ~\.ssh\InternStudio.pub)添加到InternStudio的SSH配置中
### 5.2.2 SSH端口转发
*ssh -CNg -L 6006:127.0.0.1:6006 -i  C:\Users\dukai/.ssh/InternStudio root@ssh.intern-ai.org.cn -p 33090*
其中：在本地*6006*端口监听*127.0.0.1:6006*是服务器上的web服务地址，其之前的*6006*指转发到本地的6006端口，*-p 33090*是

## 5.2 SSH配置