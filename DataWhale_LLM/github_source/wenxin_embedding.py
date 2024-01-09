
# import requests
# import json
# import os
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())



# def main():
        
#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + get_access_token(self.WENXIN_API_KEY, self.WENXIN_SECRET_KEY)
    
#     payload = json.dumps({
#         "input": ["推荐一些美食","给我讲个故事"]
#     })
#     headers = {
#         'Content-Type': 'application/json'
#     }
    
#     response = requests.request("POST", url, headers=headers, data=payload)
    
#     print(response.text)
    

# if __name__ == '__main__':
#     main()

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env
import os
import requests
import json
from dotenv import load_dotenv, find_dotenv
import pdb
import time

def get_access_token(WENXIN_API_KEY, WENXIN_SECRET_KEY):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    WENXIN_API_KEY = os.environ['WENXIN_API_KEY']
    WENXIN_SECRET_KEY = os.environ['WENXIN_SECRET_KEY']
        
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={WENXIN_API_KEY}&client_secret={WENXIN_SECRET_KEY}"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


class WenxinAIEmbeddings(Embeddings):

    wenxin_api_key: Optional[str] = None
    wenxin_secret_key: Optional[str] = None
    
    def __init__(self) -> None:
        _ = load_dotenv(find_dotenv())
        self.wenxin_api_key = os.environ['WENXIN_API_KEY']
        self.wenxin_secret_key = os.environ['WENXIN_SECRET_KEY']

        
    def _embed(self, texts: str) -> List[float]:

        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + get_access_token(self.wenxin_api_key, self.wenxin_secret_key)
    
        payload = json.dumps({"input": [texts]})
        # print(json.dumps(payload, indent=2))
        headers = {'Content-Type': 'application/json'}
        while True:
            response = requests.request("POST", url, headers=headers, data=payload)
            # print(json.dumps(response.json(), indent=2))
            # pdb.set_trace()

            if response.status_code == 200 and 'id' in response.json():
                try:
                    embeddings = response.json()['data'][0]['embedding']
                    break
                except Exception as e:
                    print(response.json())
                    print(e, e.args)

            else:
                # print(response.json().error_code, response.json().error_msg)
                break
                time.sleep(1)
                continue
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding。
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        res = []
        for idx,text in enumerate(texts):
            # print(idx)
            if idx%200==0:
                print(idx)
            res.append(self._embed(text))

        return res
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding。
        
        Args:
            text (str): 要生成 embedding 的文本。

        Return:
            List [float]: 输入文本的 embedding，一个浮点数值列表。
        """
        resp = self.embed_documents([text])
        return resp[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError(
            "Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError(
            "Please use `aembed_query`. Official does not support asynchronous requests")
 
