# -*- encoding: utf-8 -*-
# @Time : 2024/9/13 0:30
# @Author: TLIN

from typing import List
import requests
import json


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
class GPT(object):

    def __init__(self, api_url: str = 'Your_url'):
        self.api_url = api_url

    def __call__(self, prompt: str, model: str, **kwargs):

        # url = self.api_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Your Key"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        res = json.loads(response.text)

        if res["choices"][0]["message"]["content"] is not None:
            return res["choices"][0]["message"]["content"]
        else:
            return 'None'

