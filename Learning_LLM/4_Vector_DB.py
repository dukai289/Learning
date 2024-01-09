import sys
sys.path.append("..")

from langchain.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from github_source.wenxin_embedding import WenxinAIEmbeddings
from github_source.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.llms import OpenAI, HuggingFacePipeline

import os, openai, sys
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key  = os.environ['OPENAI_API_KEY']

docs = []
#pdf
file_path = "./data/pumpkin_book.pdf" # pdf
loaders = [PyMuPDFLoader(file_path)] 
for loader in loaders:
    docs.extend(loader.load())
print(file_path)
folder_path = "./data/prompt_engineering/" # md
files = os.listdir(folder_path)
loaders = []
for one_file in files:
    loader = UnstructuredMarkdownLoader(os.path.join(folder_path, one_file))
    loaders.append(loader)
for loader in loaders:
    docs.extend(loader.load())
print(folder_path)
file_path = "./data/强化学习入门指南.txt" # mp4-txt
loaders = [UnstructuredFileLoader(file_path)] #
for loader in loaders:
    docs.extend(loader.load())
print(file_path)

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)
print(len(split_docs))

# 定义 Embeddings
# embedding = OpenAIEmbeddings() 
# embedding = HuggingFaceEmbeddings(model_name='distilbert-base-uncased')
embedding = ZhipuAIEmbeddings()
# embedding = WenxinAIEmbeddings()

# 定义持久化路径
persist_directory = './vectorstores/chroma_zhipu'
# persist_directory = './vectorstores/chroma_wenxin'

# 加载数据库
vectordb = Chroma.from_documents(
    documents = split_docs,
    embedding = embedding,
    persist_directory = persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
print(vectordb)
print(f"向量库中存储的数量：{vectordb._collection.count()}")

vectordb.persist()

print('persist')