# 1. 环境相关

## 1.1 InternStudio
[InterStudio官网](https://studio.intern-ai.org.cn/console/dashboard)
### 5.1.1 创建开发机
1. 设置"开发机名称"
2. 选择并使用cuda镜像
3. 选择"资源配置"
4. 设置"运行时长"
5. 点击"立即创建"
### 1.1.2 开发机使用
+ JupyterLab
+ Terminal
+ VSCode

注意:InternLM中，在首次使用Terminal时请进入`bash`环境

## 1.2 SSH
### 1.2.1 SSH密钥配置
1. 在本地通过`ssh-keygen -t rsa`命令生成密钥对，指定文件位置`C:\Users\dukai\.ssh\InternStudio`
2. 查看生成的公钥`cat ~\.ssh\InternStudio.pub`，并将其添加到InternStudio的SSH配置中
### 1.2.2 SSH端口转发
```bash
ssh -CNg -L 6006:127.0.0.1:6006 -i  C:/Users/dukai/.ssh/InternStudio root@ssh.intern-ai.org.cn -p 33090
``` 
命令解读：在本地监听`6006`端口，并将流量转发到远程主机`ssh.intern-ai.org.cn`的`33090`端口(`-i`用来指定私钥文件)，再由远程主机转发到`127.0.0.1:6006`