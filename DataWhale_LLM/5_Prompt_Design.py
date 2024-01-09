from langchain.vectorstores import Chroma
# from lagnchain.embeddings import Embedding
from github_source.zhipuai_embedding import ZhipuAIEmbeddings

from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())


# VectorDB
embedding = ZhipuAIEmbeddings()
persist_directory = './vector_db/chroma_zhipuai_embedding'
vectordb = Chroma(
                persist_directory = persist_directory
               ,embedding_function = embedding
                )

print(f"向量库中存储的数量：{vectordb._collection.count()}")

question = r'什么是强化学习'
docs = vectordb.similarity_search(question, 3)
print(f"检索到的内容数：{len(docs)}")
for i, doc in enumerate(docs):
    print(f"检索到的第{i}个内容: \n {doc.page_content[:20]}", end="\n--------------\n")

# LLM
from langchain.chat_models import ChatOpenAI
from github_source.zhipuai_llm import ZhipuAILLM
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
llm = ZhipuAILLM()
llm.predict('你好')

# Prompt
from langchain.prompts import PromptTemplate
template = '''
    使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:
    '''
QA_CHAIN_PROMPT = PromptTemplate(input_variables = ['context', 'question'], template=template) 

# chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm
                                       ,retriever = vectordb.as_retriever()
                                       ,return_source_documents = True
                                       ,chain_type_kwargs = {'prompt': QA_CHAIN_PROMPT}
                                    )

# predict
question_1 = "什么是南瓜书？"
question_2 = "王阳明是谁？"

result = qa_chain({"query": question_1})
print("大模型+知识库后回答 question_1 的结果：")
print(result["result"])

result = qa_chain({"query": question_2})
print("大模型+知识库后回答 question_2 的结果：")
print(result["result"])


# Memory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key = 'chat_history'
                                  ,return_messages = True
                                  )

qa = ConversationalRetrievalChain.from_llm(
                llm
                ,retriever = vectordb.as_retriever()
                ,memory = memory
                )
question = "我可以学习到关于强化学习的知识吗？"
result = qa({"question": question})
print(result['answer'])
question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
print(result['answer'])
