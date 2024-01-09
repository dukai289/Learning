import sys
sys.path.append("..")

from dotenv import find_dotenv, load_dotenv
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from github_source.zhipuai_embedding import ZhipuAIEmbeddings
from github_source.wenxin_embedding import WenxinAIEmbeddings
from github_source.zhipuai_llm import ZhipuAILLM
from github_source.wenxin_llm import Wenxin_LLM
from langchain.chains import LLMChain ,RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json

_ = load_dotenv(find_dotenv())

llm_model = 'zhipuai'
llm_model = 'wenxin'

if llm_model == 'wenxin':
    embedding_function = WenxinAIEmbeddings()
    persist_directory = '../vectorstores/chroma_wenxin'
    llm = Wenxin_LLM()
elif llm_model == 'zhipuai':
    embedding_function = ZhipuAIEmbeddings()
    persist_directory = '../vectorstores/chroma_zhipu'
    llm = ZhipuAILLM(model="chatglm_std", temperature=0)

# vectordb

# 
# e = embedding_function._embed('测试数据')
# print(e)

# 
# persist_directory = './vector_db/chroma_zhipuai_embedding'
vectorstore = Chroma(persist_directory = persist_directory,embedding_function = embedding_function)
print(f"向量库中存储的数量：{vectorstore._collection.count()}")


# llm

# llm = ZhipuAILLM()
# result = llm.predict('你好')
# print(result)

# prompt
template = '''
    根据上下文和会话历史回答问题。
    如果你不知道答案，就说你不知道，不要试图编造答。
    回答不超过100字。如果回答过长，请分为各个小点。

    上下文：{context}
    问题：{question}
    '''
prompt = PromptTemplate(input_variables=['context','question'],template=template)

# memory
memory = ConversationBufferMemory(memory_key = 'history', return_messages = True)

# chain
# llm_chain = LLMChain(llm = llm ,memory = memory ,prompt = prompt)
qa_chain = RetrievalQA.from_chain_type(
                    llm = llm   
                    ,retriever = vectorstore.as_retriever()
                    ,memory = ConversationBufferMemory()
                   ,chain_type_kwargs={"prompt": prompt}
                   )

# apply
def qa(question):
    if llm_model == 'zhipuai':
        query = {'query': question}
        result = qa_chain(query)
        return result['answer']
    elif llm_model == 'wenxin':
        result = qa_chain(question)
        return result['result']

dialog_history = []
questions = ['强化学习包含哪些知识？', '怎样学习这些知识？']
for idx, question in enumerate(questions):
    # dialog_history.append(question)
    # inp = {'question': question, 'history': ''.join(dialog_history)}
    inp = {'question': question}
    # answer = qa(inp)
    answer = qa(question)
    print(idx+1)
    print('\t', {'question': question})
    print('\t', {'answer': answer})


from transformers import pipeline, AutoModel

pipe = pipeline()
model = AutoModel.from_pretrained()