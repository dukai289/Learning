import json
import time
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Field, model_validator
# from pydantic.functional_validators import model_validator
from dotenv import find_dotenv, load_dotenv
import os
import pdb
_ = load_dotenv(find_dotenv())

def get_access_token(api_key : str, secret_key : str):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    # api_key = os.environ['WENXIN_API_KEY']
    # secret_key = os.environ['WENXIN_SECRET_KEY']
    # 指定网址
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # print(url)
    # 设置 POST 访问
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # 通过 POST 访问获取账户对应的 access_token
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

# 继承自 langchain.llms.base.LLM
class Wenxin_LLM(LLM):
    # 原生接口地址
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型
    model_name: str = Field(default="ERNIE-Bot-turbo", alias="model")
    # 访问时延上限
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    # 温度系数
    temperature: float = 0.01
    # API_Key
    api_key: str = os.environ['WENXIN_API_KEY']
    # Secret_Key
    secret_key : str = os.environ['WENXIN_SECRET_KEY']
    # access_token
    access_token: str = get_access_token(api_key,secret_key)
    # 必备的可选参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "wenxin"
    
    def init_access_token(self):
        if self.api_key != None and self.secret_key != None:
            # 两个 Key 均非空才可以获取 access_token
            try:
                self.access_token = get_access_token(self.api_key, self.secret_key)
            except Exception as e:
                print(e)
                print("获取 access_token 失败，请检查 Key")
        else:
            print("API_Key 或 Secret_Key 为空，请检查 Key")
        

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any):
        # 除 prompt 参数外，其他参数并没有被用到，但当我们通过 LangChain 调用时会传入这些参数，因此必须设置
        # 如果 access_token 为空，初始化 access_token
        if self.access_token == None:
            self.init_access_token()
        # API 调用 url
        # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}".format(self.access_token)
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={}".format(self.access_token) # ERNIE-Bot
        # 配置 POST 参数
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",# user prompt
                    "content": "{}".format(prompt)# 输入的 prompt
                }
            ],
            'temperature' : self.temperature
        })
        headers = {
            'Content-Type': 'application/json'
        }
        # 发起请求
        response = requests.request("POST", url, headers=headers, data=payload, timeout=self.request_timeout)
        # pdb.set_trace()
        if response.status_code == 200:
            # 返回的是一个 Json 字符串
            js = json.loads(response.text)
            return js["result"]
        else:
            return "请求失败"
    # # 首先定义一个返回默认参数的方法
    # @property
    # def _default_params(self) -> Dict[str, Any]:
    #     """获取调用Ennie API的默认参数。"""
    #     normal_params = {
    #         "temperature": self.temperature,
    #         "request_timeout": self.request_timeout,
    #         }
    #     return {**normal_params}


    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """Get the identifying parameters."""
    #     return {**{"model_name": self.model_name}, **self._default_params}
