# -*- encoding: utf-8 -*-
# @Time : 2024/9/13 0:33

from wikirag.qageneration import AutoQA
# from wikirag.transformation import Transfer
from llm.gpt import GPT
from llm.llama3 import Llama3
from wikirag.db_utils import dbutils
import pandas
import sqlite3
import pandas as pd
import json
import os

# RAG imports
from typing import List, Dict
from rag import build_hierarchical_rag_with_openai
from wiki_utils import wikiutils

def agent(model, prompt):
    gpt = GPT()
    llama = Llama3()
    # prompt = prompt
    if model in ['GPT-4', 'GPT-3.5']:
        return gpt(prompt, model)
    else:
        return llama.run(prompt)

def check(question, groundrtrue, answer):
    prompt = '''
     The question is {}, the true answer is {},the given answer is{}.if the given answer meet the true answer,output 'True', if not meet, output'False'.just output the the 'True' or 'False', no other words.
    '''.format(question, groundrtrue, answer)

    return agent('GPT-3.5', prompt)

def gpt_eval(path, outfile='gpt_out.jsonl', qfile='MEBench_test.jsonl', model='GPT-4'):
    idx = 0
    cout_t = 0
    fout_qa = open(os.path.join(path, outfile), 'a')
    with open(os.path.join(path, qfile), 'r') as file:
        # Read each line in the file
        for line in file:
            # Convert the JSON string to a Python dictionary
            data = json.loads(line)
            # sqll = data['sql'].split(';')[0]
            # table_n = data['topic'].replace(' ', '').replace('-', '')
            # qtype = data['type']

            try:
                topic = data.get('topic', '')
                edge = data.get('edge', '')
                _docs = build_docs_from_topic_edge(topic, edge)
                _corpus_titles = [d.get('title', '') for d in _docs]
            except Exception:
                _corpus_titles = []

            prompt = '''
                The question is {}, answers the question, just output the the answer, no other words.
                '''.format(data['question'])
            answe = agent(model, prompt)
            checkout = check(data['question'], data['goldanswer'], answe)
            if checkout == 'True':
                cout_t += 1
            ques_i = {'qid': idx, 'topic': data['topic'], 'edge': data['edge'], 'class': data['class'],
                      'hop': data['hop'], 'properties': data['properties'], 'type': data['type'],
                      'question': data['question'], 'sql': data['sql'], 'answer': answe, 'goldanswer': data['goldanswer'],'true': checkout,
                      'corpus_titles': _corpus_titles}
            idx += 1
            fout_qa.write(json.dumps(ques_i) + '\n')

    fout_qa.close()
    acc = cout_t / idx
    print('The acc of GPT: {}'.format(acc))

def llama_eval(path, outfile='llama_out.jsonl', qfile='MEBench_test.jsonl', model='llama3'):
    idx = 0
    cout_t = 0
    fout_qa = open(os.path.join(path, outfile), 'a')
    with open(os.path.join(path, qfile), 'r') as file:
        # Read each line in the file
        for line in file:
            # Convert the JSON string to a Python dictionary
            data = json.loads(line)

            try:
                topic = data.get('topic', '')
                edge = data.get('edge', '')
                _docs = build_docs_from_topic_edge(topic, edge)
                _corpus_titles = [d.get('title', '') for d in _docs]
            except Exception:
                _corpus_titles = []

            prompt = '''
                The question is {}, answers the question, just output the the answer, no other words.
                '''.format(data['question'])
            answe = agent(model, prompt)

            checkout = check(data['question'], data['goldanswer'], answe)
            if checkout == 'True':
                cout_t += 1
            ques_i = {'qid': idx, 'topic': data['topic'], 'edge': data['edge'], 'class': data['class'],
                      'hop': data['hop'], 'properties': data['properties'], 'type': data['type'],
                      'question': data['question'], 'sql': data['sql'], 'answer': answe, 'goldanswer': data['goldanswer'],'true': checkout,
                      'corpus_titles': _corpus_titles}
            idx += 1
            fout_qa.write(json.dumps(ques_i) + '\n')

    fout_qa.close()
    acc = cout_t/idx
    print('The acc of llama3: {}'.format(acc))


def _load_docs_from_jsonl(jsonl_path: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not os.path.exists(jsonl_path):
        return docs
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            doc_id = d.get('id') or d.get('doc_id') or str(idx)
            title = d.get('title') or d.get('topic') or None
            text = d.get('text') or d.get('content') or ''
            if not text:
                continue
            docs.append({'id': str(doc_id), 'title': title, 'text': text})
    return docs


def build_docs_from_topic_edge(topic: str, edge: str) -> List[Dict[str, str]]:
   
    wu = wikiutils()
    try:
        topic_id = wu.get_wikidata_id(topic)
        if not topic_id:
            return []
        props = wu.search_wikidata_property(edge)
        if not props:
            return []
        property_id = props[0][0]
        titles = wu.get_entities_wikipedia_titles(property_id, topic_id)
        docs: List[Dict[str, str]] = []
        for i, t in enumerate(titles):
            para = wu.get_wikipedia_first_paragraph(t)
            if not para or para.strip() == "":
                continue
            docs.append({'id': f'wikidoc_{i}', 'title': t, 'text': para})
        return docs
    except Exception:
        return []


def gpt_eval_rag(
        path: str,
        outfile: str = 'gpt_rag_out.jsonl',
        qfile: str = 'MEBench_test.jsonl',
        model: str = 'GPT-4',
        top_k: int = 5,
    ):
    idx = 0
    cout_t = 0
    fout_qa = open(os.path.join(path, outfile), 'a', encoding='utf-8')
    with open(os.path.join(path, qfile), 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            question = data['question']
            topic = data.get('topic', '')
            edge = data.get('edge', '')

            
            docs = build_docs_from_topic_edge(topic, edge)
            if len(docs) == 0:
                best_context = ''
                best_doc_id = None
                best_chunk_ids: List[str] = []
            else:
                rag = build_hierarchical_rag_with_openai(docs, openai_api_key=os.getenv('OPENAI_API_KEY'))
                results = rag.retrieve(question, top_k_per_doc=top_k)
                best_doc_id = None
                best_context = ''
                best_chunk_ids = []
                best_len = -1
                for doc_id, ctx in results.items():
                    if len(ctx.context) > best_len:
                        best_len = len(ctx.context)
                        best_doc_id = doc_id
                        best_context = ctx.context
                        best_chunk_ids = ctx.chunk_ids

            # Retrieve per document and concatenate contexts by simple score: choose the best doc context
            best_doc_id = None
            best_context = ''
            best_chunk_ids: List[str] = []
            best_len = -1
            results = rag.retrieve(question, top_k_per_doc=top_k)
            for doc_id, ctx in results.items():
                if len(ctx.context) > best_len:
                    best_len = len(ctx.context)
                    best_doc_id = doc_id
                    best_context = ctx.context
                    best_chunk_ids = ctx.chunk_ids

            prompt = f"""
You are given the following context retrieved from relevant documents. Use it to answer the question concisely.
Context:
{best_context}

Question: {question}
Only output the final answer.
""".strip()

            answe = agent(model, prompt)
            checkout = check(question, data['goldanswer'], answe)
            if checkout == 'True':
                cout_t += 1
            ques_i = {
                'qid': idx,
                'topic': data['topic'],
                'edge': data['edge'],
                'class': data['class'],
                'hop': data['hop'],
                'properties': data['properties'],
                'type': data['type'],
                'question': question,
                'sql': data['sql'],
                'answer': answe,
                'goldanswer': data['goldanswer'],
                'true': checkout,
                'rag_doc_id': best_doc_id,
                'rag_chunk_ids': best_chunk_ids,
            }
            idx += 1
            fout_qa.write(json.dumps(ques_i, ensure_ascii=False) + '\n')

    fout_qa.close()
    acc = cout_t / idx if idx else 0
    print('The acc of GPT with RAG: {}'.format(acc))


def llama_eval_rag(
        path: str,
        outfile: str = 'llama_rag_out.jsonl',
        qfile: str = 'MEBench_test.jsonl',
        model: str = 'llama3',
        top_k: int = 5,
    ):
    idx = 0
    cout_t = 0
    fout_qa = open(os.path.join(path, outfile), 'a', encoding='utf-8')
    with open(os.path.join(path, qfile), 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            question = data['question']
            topic = data.get('topic', '')
            edge = data.get('edge', '')

           
            docs = build_docs_from_topic_edge(topic, edge)
            if len(docs) == 0:
                best_context = ''
                best_doc_id = None
                best_chunk_ids: List[str] = []
            else:
                rag = build_hierarchical_rag_with_openai(docs, openai_api_key=os.getenv('OPENAI_API_KEY'))
                results = rag.retrieve(question, top_k_per_doc=top_k)
                best_doc_id = None
                best_context = ''
                best_chunk_ids = []
                best_len = -1
                for doc_id, ctx in results.items():
                    if len(ctx.context) > best_len:
                        best_len = len(ctx.context)
                        best_doc_id = doc_id
                        best_context = ctx.context
                        best_chunk_ids = ctx.chunk_ids

            prompt = f"""
You are given the following context retrieved from relevant documents. Use it to answer the question concisely.
Context:
{best_context}

Question: {question}
Only output the final answer.
""".strip()

            answe = agent(model, prompt)
            checkout = check(question, data['goldanswer'], answe)
            if checkout == 'True':
                cout_t += 1
            ques_i = {
                'qid': idx,
                'topic': data['topic'],
                'edge': data['edge'],
                'class': data['class'],
                'hop': data['hop'],
                'properties': data['properties'],
                'type': data['type'],
                'question': question,
                'sql': data['sql'],
                'answer': answe,
                'goldanswer': data['goldanswer'],
                'true': checkout,
                'rag_doc_id': best_doc_id,
                'rag_chunk_ids': best_chunk_ids,
            }
            idx += 1
            fout_qa.write(json.dumps(ques_i, ensure_ascii=False) + '\n')

    fout_qa.close()
    acc = cout_t/idx if idx else 0
    print('The acc of llama3 with RAG: {}'.format(acc))


def main():
    path = 'benchmark'

    gpt_eval(path)

    llama_eval(path)

    # RAG-based evaluations (uncomment to run)
    gpt_eval_rag(path)
    llama_eval_rag(path)

    # sgraph(path)


if __name__ == '__main__':
    main()