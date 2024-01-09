import os
import pdb
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

#############################################################################################################
# 1. chatGPT： # https://atlassc.net/2023/04/25/azure-openai-service
'''
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# 1.1 原生接口
completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo"
               ,message=[
                        {'role':'system', 'content':'You are a helpful assistant.'}
                    #    ,{'role':'assistance', 'content':'Hello!'}
                       ,{'role':'user', 'content':'Hello!'}
                        ]
                    )
print(completion.choices[0]['message']['content'])

def get_completion(prompt, model='gpt-3.5-turbo', temperature=0):
    message = [{'role':'user', 'content':prompt}]
    response = openai.ChatCompletion.create(
                    model=model
                    ,message = message
                    ,temperature = temperature
                    )
    return response.choices[0]['message']['content']

## 1.2 Langchain接口
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

template_string = """Translate the text \
that is delimited by triple backticks \
into a Chinses. \
text: ```{text}```
"""
chat_template = ChatPromptTemplate.from_template(template_string)
text = 'Today is a nice day.'
message = chat_template.format(text=text)
print(message)
chat = ChatOpenAI(temperature=0.0)
response = chat(message)
print(response)
'''
#############################################################################################################


# 2. 文心一言
## 2.1 access_token
import requests
import json
def get_access_token():
    api_key = os.getenv("WENXIN_API_KEY")
    secret_key = os.getenv("WENXIN_SECRET_KEY")
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}".format(api_key=api_key, secret_key=secret_key)
    payload = json.dumps('')
    headers = {'Content_Type': 'application/json', 'Accept': 'application/json'}
    response = requests.request('POST', url, headers=headers, data=payload)
    return response.json()['access_token']
## 2.2 原生接口
def get_wenxin(prompt, temperature=0.1):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={access_token}".format(access_token=get_access_token())
    payload = json.dumps({"message": [{'role':'user', 'content':'{}'.format(prompt)}]
                          ,'temperature': temperature
                          })
    headers = {'Content_type': 'application/json'}
    response = requests.request('POST', url, headers=headers, data=payload)
    pdb.set_trace()
    js = json.loads(response.text)
    print(js['result'])
prompt = '你好'
# get_wenxin(prompt)
## 2.2 Langchain接口
from wenxin_llm import Wenxin_LLM
wenxin_api_key = os.environ["WENXIN_API_KEY"]
wenxin_secret_key = os.environ["WENXIN_SECRET_KEY"]
llm = Wenxin_LLM(api_key=wenxin_api_key, secret_key=wenxin_secret_key)
llm("你好")

#############################################################################################################
# 3. 讯飞星火
'''
import SparkApi
appid = os.environ["SPARK_APPID"]
spark_api_key = os.environ["SPARK_API_KEY"]
spark_secret_key = os.environ["SPARK_SECRET_KEY"]
domain = 'general'
Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"
def getText(role: str, content: str, text: list =[]):
    jsoncon = {}
    jsoncon['role'] = role
    jsoncon['content'] = content
    text.append(jsoncon)
    return text
question = getText('user', '你好')
print(question)

response = SparkApi.main(appid, spark_api_key, spark_secret_key, Spark_url, domain, question)
print(response)
'''