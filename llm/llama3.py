# -*- encoding: utf-8 -*-
# @Time : 2024/6/28 15:32
# @Author: TLIN

import torch
from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
# from modelscope import snapshot_download
from dataclasses import dataclass
from typing import Dict
import copy


@dataclass
class Template:
    template_name: str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str


template_dict: Dict[str, Template] = dict()


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )


#
register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}\n',  # \n\n{content}<|eot_id|>\n
    system="",
    stop_word='<|eot_id|>'
)

class Llama3(object):

    def __init__(self, adapter='', model="/root/EmoLLM/xtuner_config/merged_Llama3_8b_instruct", **kwargs):
        # download model in openxlab
        # download(model_repo='MrCat/Meta-Llama-3-8B-Instruct',
        #        output='MrCat/Meta-Llama-3-8B-Instruct')
        # model_name_or_path = 'MrCat/Meta-Llama-3-8B-Instruct'

        # # download model in modelscope
        # model_name_or_path = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct',
        #                                        cache_dir='LLM-Research/Meta-Llama-3-8B-Instruct')

        # offline model
        model_name_or_path = model  # "/root/EmoLLM/xtuner_config/merged_Llama3_8b_instruct"
        template_name = 'llama3'
        adapter_name_or_path = adapter

        self.template = template_dict[template_name]

        load_in_4bit = False

        # load model
        print(f'Loading model from: {model_name_or_path}')
        print(f'adapter_name_or_path: {adapter_name_or_path}')
        self.model = load_model(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            adapter_name_or_path=adapter_name_or_path
        ).eval()
        self.tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
        if self.template.stop_word is None:
            self.template.stop_word = self.tokenizer.eos_token
        self.stop_token_id = self.tokenizer.encode(self.template.stop_word, add_special_tokens=True)
        # assert len(stop_token_id) == 1
        self.stop_token_id = self.stop_token_id[0]

    def run(self, prompt):
        max_new_tokens = 500  
        top_p = 0.9
        temperature = 0.6 
        repetition_penalty = 1.1  

        history = []
        input_ids = build_prompt(self.tokenizer, self.template, prompt, copy.deepcopy(history), system=None).to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=self.stop_token_id, pad_token_id=self.tokenizer.eos_token_id
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = self.tokenizer.decode(outputs)
        response = response.strip().replace(self.template.stop_word, "").strip()


        # history.append({"role": 'user', 'message': query})
        # history.append({"role": 'assistant', 'message': response})
        #
       
        # if len(history) > 12:
        #     history = history[:-12]
        return response.replace('\n', '').replace('<|start_header_id|>', '').replace('assistant<|end_header_id|>', '')


def load_model(model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=quantization_config
    )

    # load adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    return model


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    history.append({"role": 'user', 'message': query})
    input_ids = []

    if system_format is not None:
        if system is not None:
            system_text = system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)

    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids

