# 1. 主题
 - RAG

# 2. 大纲
## 2.1 要素
 - 开发机
 - Embedding工具
 - LLM
 - RAG框架
 - 数据集

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
工作目录 `~/RAG-cq`
```bash
bash
mkdir ~/RAG-cq &&　cd  ~/RAG-cq
```
conda环境 `RAG-cq`
```bash
/root/share/install_conda_env_internlm_base.sh RAG-cq
conda activate RAG-cq

python -m pip install --upgrade pip
pip install modelscope==1.9.5 transformers==4.35.2 streamlit==1.24.0 sentencepiece==0.1.99 accelerate==0.24.1
```


返回工作目录 `~/RAG-cq`
```bash
cd ~/RAG-cq
```

# 4. Embedding
安装开源词向量模型 Sentence Transformer
```bash
conda install huggingface_hub[cli]
# pip install -U "huggingface_hub[cli]"
# pip install huggingface-cli
# pip install -U huggingface_hub
```
在 `/root/data` 目录下新建python文件 `download_hf.py`
```bash
touch /root/data/download_hf.py && vim /root/data/download_hf.py
```
内容如下：
```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 设置镜像

# 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/data/model/sentence-transformer')
```
运行下载脚本
```bash
python /root/data/download_hf.py
```

NLTK配置
```bash
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

# 5. LLM
从 `ModelScope` 下载 `ZhipuAI/chatglm3-6b` 模型  
新建 `download_model.py`
```bash
touch download_model.py && vim download_model.py
```
内容如下：
```python
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/chatglm3-6b', cache_dir='/root/data/model', revision='v1.0.0')
```
运行下载脚本 
```bash
python download_model.py
```
查看下载好的LLM
```bash
ll /root/data/model/ZhipuAI/chatglm3-6b
```

# 6. RAG框架 `Langchain`
安装 `Langchain`
```
pip install langchain==0.0.292 gradio==4.4.0 chromadb==0.4.15 sentence-transformers==2.2.2 unstructured==0.10.30 markdown==3.3.7
```

# 7. 数据集
上传数据到 `/root/RAG-cq/data` 目录
```bash
mkdir /root/RAG-cq/data
# 上传数据
ll /root/RAG-cq/data
```

# 8. 构建向量数据库
```bash
touch build_db.py && vim build_db.py
```
```python
# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import os

# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md") or filename.endswith(".txt") or filename.endswith(".docx"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
    return file_list

# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'docx':
            loader = UnstructuredWordDocumentLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs

# 目标文件夹
tar_dir = [
    "/root/RAG-cq/data",
    ]

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

# 构建向量数据库
# 定义持久化路径
persist_directory = '/root/RAG-cq/data_base/vector_cq'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()

print('向量数据库构建完成')
```
运行文件
```bash
python build_db.py
```

# 9. ChatGLM接入Langchain
编写`LLM.py`脚本 自定义`Langchain`的LLM子类(重写方法 `__init__` 和 `_call`)
```bash
touch LLM.py && vim LLM.py
```
```python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

class ChatGLM_LLM(LLM):
    tokenizer : AutoTokenizer = None
    # model: AutoModelForCausalLM = None
    model: AutoModel = None

    def __init__(self, model_path :str):
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self
              , prompt : str
              , stop: Optional[List[str]] = None
              , run_manager: Optional[CallbackManagerForLLMRun] = None
              , **kwargs: Any
             ):
        # 重写调用函数
        system_prompt = """
        你是一个知识助手，请在认真思考后得出回答。如果不清楚或者不知道，请诚实回答。
        """

        history = [{'role': 'system', 'content': system_prompt}]
        
        response, history = self.model.chat(self.tokenizer, prompt , history=history, top_p=0.05, temperature=0.05)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"
```

# 10. 探索：构建检索问答链
本节探索内容都在JupyterLab中进行，注意仍在`/root/RAG-cq`目录下  
将conda环境加入`JupyterLab`
```bash
lab add RAG-cq
```
## 10.1 加载向量数据库
```
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os

# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

# 向量数据库持久化路径
persist_directory = 'data_base/vector_cq'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)
```

## 10.2 实例化自定义 LLM 与 Prompt Template
实例化
```python
from LLM import ChatGLM_LLM
llm = ChatGLM_LLM(model_path = "/root/data/model/ZhipuAI/chatglm3-6b")
llm.predict("你是谁")
```
构建`PromptTemplate`
```python
from langchain.prompts import PromptTemplate

# 我们所构造的 Prompt 模板
template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答你不知道。
有用的回答:
"""

# 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
```

## 10.3 构建检索问答链
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
```
对比`LLM`与`RAG`
```python
# 检索问答链回答效果
question = "什么是InternLM"
result = qa_chain({"query": question})
print("检索问答链回答 question 的结果：")
print(result["result"])

# 仅 LLM 回答效果
result_2 = llm(question)
print("大模型回答 question 的结果：")
print(result_2)
```

# 11. 部署
`run_gradio.py`
```bash
touch run_gradio.py && vim run_gradio.py
```
```python
# 1. 定义 load_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import ChatGLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_cq'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    # 加载自定义 LLM
    llm = ChatGLM_LLM(model_path = "/root/data/model/ZhipuAI/chatglm3-6b")

    # 定义一个 Prompt Template
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain

# 2. 定义Model_center
class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history

# 3. 使用gradio搭建UI
import gradio as gr

# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>长轻知识库</center></h1>
                <center>ChatGLM实现RAG</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
demo.launch()
```

运行
```bash
python run_gradio.py
```

# 12. 端口转发
```bash
ssh -CNg -L 6006:127.0.0.1:6006 -i  C:/Users/dukai/.ssh/InternStudio root@ssh.intern-ai.org.cn -p 33090
```

# 13. 问题定位
以
```python
question = "对于没有血脂肥胖问题的、没有胰岛受损的人群用这个方案的时候怎么和客户介绍这个阶段的效果是适合她的呢？"
```
为例，
原始文档中的回答为：
```text
均衡抗炎三个月的安排，就前面一个月是以炎症、胰岛和肠道来打底，中间这个月，集中在肝脏的修复，包括胆囊的胆汁促进流动的修复。肝脏是整个代谢的一个枢纽。
早晨轻微的宿醉感（患者自述的症状），其实就是肝脏受损。前面第一个月，是在为肝脏打基础的，优先去把炎症解决、胰岛素抵抗解决、肠道的问题解决，再去做脂肪肝或者是肝脏的修复，肝脏解决了之后，再去解决需要的终极结果就是血管健康的一个问题，整体来说从多个角度去给患者讲解。
```

RAG的回答结果为：
```text
对于没有血脂肥胖问题和胰岛受损的人群，这个方案的阶段主要是以抗炎为主，肝脏、胰岛和肠道方面的修复。在肝脏修复阶段，需要关注胆囊的胆汁流动的修复。这个阶段可能会有一些轻微的宿醉感，其实是肝脏受损的表现。在能量供应方面，方案要求执行168饮食原则，对于血糖异常的学员也要尽量坚持。如果学员不想减重，可以灵活增加食材的量，优选蔬菜和优质蛋白的食物。在执行过程中，需要注意避免能量过低，因为低能量会限制糖脂供能切换的效果。
```
而 `result['source_documents']` 的内容为：
```text
[Document(page_content='4、对于没有血脂肥胖问题的、没有胰岛受损的人群用这个方案的时候怎么和客户介绍这个阶段的效果是适合她的呢？\n\n       均衡抗炎三个月的安排，就前面一个月是以炎症、胰岛和肠道来打底，中间这个月，集中在肝脏的修复，包括胆囊的胆汁促进流动的修复。肝脏是整个代谢的一个枢纽。\n\n早晨轻微的宿醉感（患者自述的症状），其实就是肝脏受损。前面第一个月，是在为肝脏打基础的，优先去把炎症解决、胰岛素抵抗解决、肠道的问题解决，再去做脂肪肝或者是肝脏的修复，肝脏解决了之后，再去解决需要的终极结果就是血管健康的一个问题，整体来说从多个角度去给患者讲解。\n\n是否可以理解均衡抗炎方案为温和方案的升级版，它和4816慢线有什么区别？\n\n 均衡抗炎不是温和版方案的升级版，也不是一个降级版，是新设计的一条线，没有相关的关系，是新的概念。\n\n医学指标如何评估是胰岛素抵抗，让学员清楚的知道有这个问题？\n\n胰岛素抵抗指数=空腹胰岛素(mu/l)*空腹血糖(mmol/l)÷22.5，大于2.69可判断胰岛素抵抗，这种方法可以大概反映胰岛素抵抗的情况（不是金标准）。\n医院测量的指标：空腹血糖和空腹胰岛素水平，即可判断。', metadata={'source': '/root/RAG-cq/data/知识库内容支持--营养科学部.docx'}),
 Document(page_content='2、本阶段执行建议168饮食原则，血糖异常的学员也要尽量坚持吗？是不是建议学员监测血糖情况，灵活调整降糖药，有些学员测血糖不方便，不想频繁扎手，是否建议执行方案的时候先停掉降糖药？\n\n 我们的碳水没有降的太多，血糖异常的人看病程的时长，同时也要看营养食疗师是否建议客户执行168。血糖越到中晚期，要建议少吃多餐，就不建议执行168了，因为他的血糖要靠吃饭和运动来维持。如果不吃饭，血糖波动会很大，现在给到的是80到120克碳水，这一周先是80克，然后100克，最后是120克循序渐进。\n\n阿卡波糖类的药、胰岛素的话可以适当的调整，因为能量太低了（具体还是要多监测患者的血糖情况）。\n\n3、如果学员不想减重，是不是可以食材的量可以灵活增加，优选蔬菜和优质蛋白的食物增加吗？\n\n       这周方案是不涉及系数，是一个固定的能量的餐单。不想减重，执行5到7天的方案就行了，是1200千卡。如果再想多吃，也是建议可以随时再加一点。但是就没有方案效果的作用了。但可以参考餐单去吃，只是没有太大的意义，相当于放弃这个方案了。\n\n4、本阶段不想减重及血糖高的学员执行过程中有什么特别需要注意的点？-月萍', metadata={'source': '/root/RAG-cq/data/知识库内容支持--营养科学部.docx'}),
 Document(page_content='有肾脏功能异常者不适合走均衡抗炎方案\n\n2、胰岛平衡，阶段后两天没看到有糖脂供能切换的饮食训练，这个是否会影响到学员的调理效果和体征感受？\n\n       标准方案第二阶段是客户瘦的比较快的一个阶段而均衡饮食的代谢训练会变得更柔和一点。所以第一个月并没有一个强度比较大的糖脂供能切换，到了第五周才会有糖脂供能切换（代谢激活）。做细胞净化的时候，其实也是糖脂供能的切换，强度会更大。肝糖原会在24-48小时之内消耗完。到了5加2的饮食模式时，强度是连续两天去做断食，切换的效果基本上是在两天完成糖脂切换。均衡抗炎方案要到第五周才给到的代谢功能改变，没有给到低碳水，碳水占比不低于40%。所以实际上是需要一周来完成糖脂供能切换，就没有安排在第二周。\n       均衡抗炎，还是希望第一周通过低GI饮食、胰岛改善和肠道的改善，来帮助客户改善他的胰岛抵抗，提高他的胰岛的敏感性，让他的血糖更平稳，是从这个角度做一个改善的，没有强度很大的糖脂供能切换训练。只能说限制了能量和低GI，此时胰岛素水平肯定比之前分泌要低。', metadata={'source': '/root/RAG-cq/data/知识库内容支持--营养科学部.docx'}),
 Document(page_content='方案的功效或作用\n\n1、这里是通过高蛋白提供细胞充足的代谢原料，从而增加代谢排毒和缓解胰岛素抵抗对吗？\n\n       首先这个方案不算高蛋白饮食方案。高蛋白增加代谢排毒，大部分是用在肝脏方案里的，在肝脏方案需要用到高蛋白饮食。在胰岛修复里又要用到一个适量蛋白或者是偏低一点的蛋白摄入量。\n\n高蛋白是给细胞提供修复、代谢的原料，但不是这个方案的主要目标。这个方案上只是保证了蛋白没有过低。能量是800千卡的话，蛋白质是800大概是60克左右，对于大多数人，60克不算是高蛋白饮食，所以相对来说还是属于一个均衡饮食的范畴。糖脂供能切换的时候会用到高蛋白，需要保证一定的蛋白的供应的量，保护肌肉和稳定代。此阶段还没有涉及到代谢排毒，因为代谢排毒，只给到高蛋白是不够的，要有排毒的营养因素加进去才可以。\n\n2、如果不能按照方案来吃，需保持餐盘比大概保持2:1:1\n\n餐盘比大概是211，蛋白的量可以不用太低\n\n食材的选择和作用\n\n1、本阶段餐单食材戒除了麸质，推荐的食物中有钢切燕麦、玉米棒，这些食材也含有麸质，是因为含麸质的量少所以可以选择吗？', metadata={'source': '/root/RAG-cq/data/知识库内容支持--营养科学部.docx'})]
```

# 14. 解决办法

## 14.1 LLM侧
在调用 `LLM.chat` 时设置 `top_p=0.05, temperature=0.05`  
发现效果一般

## 14.2 prompt侧
将文档内容加入到prompt中
```python
file = r'/root/RAG-cq/data/知识库内容.xlsx'

import pandas as pd
import json

df = pd.read_excel(file)
df.head()

prompt_data = []
for idx,row in df.iterrows():
    input_row = '关于`{}`的`{}`问题,{}?'.format(row['info_1'], row['info_2'], row['Q'])
    conversation = [
        {'role':'user'
        ,'content': input_row
        }
       ,{'role': 'assistant'
         ,'metadata': ''
         ,'content': row['A']
        }
       ]
    prompt_data.extend(conversation)
len(prompt_data)


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
model_dir = "/root/data/model/ZhipuAI/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model = model.eval()

question = "对于没有血脂肥胖问题的、没有胰岛受损的人群用这个方案的时候怎么和客户介绍这个阶段的效果是适合她的呢？"

response, history = model.chat(tokenizer, prompt_data , history=output_data)
response
```
遇到的问题在于：
- 文档转化为prompt后，对话轮数超出限制

## 14.3 检索侧
