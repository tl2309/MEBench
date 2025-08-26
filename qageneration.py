# -*- encoding: utf-8 -*-

from llm.gpt import GPT

import pandas
import sqlite3
from tqdm import tqdm

import os
import pandas as pd
import json
from wikirag.wiki_utils import wikiutils
import random


def sqltool(sql_query):
    # sql_table(file_path, header)
    conn = sqlite3.connect('document_store.db')
    cursor = conn.cursor()
    columns = []
    # sql_query = "SELECT * FROM your_table_name;"
    try:

        cursor.execute("SELECT name FROM IEEEFellow;")
        rows = cursor.fetchall()

        # Print the list of tables
        print("Tables in the database:")
        for table in rows:
            print(table[0])

        cursor.execute(sql_query)
        rows = cursor.fetchall()
        # print('answer is:', rows)
        if cursor.description != None:
            columns = [description[0] for description in cursor.description]
        conn.commit()
        conn.close()

        print(columns)

        return columns, rows

    except sqlite3.Error as e:
        print("An error occurred:", e)
        return []
    # with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(columns)
    #     writer.writerows(rows)
    #
    # table_name = 'faculty'
    # cursor.execute(f'DROP TABLE IF EXISTS {table_name}')


def csv_sqltool(table, sql_query):
    try:
        csv_file_path = "outputcsv/{}.csv".format(table)
        df = pd.read_csv(csv_file_path)

        # Create a connection to a new in-memory SQLite database
        conn = sqlite3.connect(':memory:')

        # Copy the DataFrame data into a new SQLite table
        df.to_sql(table, conn, if_exists='replace', index=False)

        # Execute the SQL query
        result_df = pd.read_sql_query(sql_query, conn)

        # Print the result DataFrame
        # print('reults = ', result_df)
        columns = ', '.join(result_df.columns)
        rows = []
        for row in result_df.itertuples(index=False, name=None):
            # print('row = ',row)
            rows.append(row)
        # print(rows)
        return columns, rows
    except pandas.errors.DatabaseError as e:
        print("Database Error:", e)
        print("An error occurred:", e)
        return 'Wrong'


class AutoQA(object):
    def __init__(self, path='Stran', **kwargs):
        self.path = path
        self.wikiutils = wikiutils()
        self.GPT = GPT()

    def gen_m_query(self, qfile='multi_entities_query.jsonl', efile='entities_first_paragraph.jsonl'):
        fout_queries = open(os.path.join(self.path, qfile), 'a')
        fout_entity = open(os.path.join(self.path, efile), 'a')

        topics = [
            "Ivy League"
            # "Russell Group",
            # "Group of Eight"
        ]  # , "Cities of the World", "Presidents of the US", "Chemical Elements", "Summer Olympic Games", "Nobel Prize in Chemistry"]
        properties = ["name, start year, president, location, staff number"]
        # ,
        #             "population, geographic coordinates, altitude, GDP",
        #             "term lengths, political parties, vice-presidents, birth states, previous occupations",
        #             "atomic number, atomic mass, boiling point, melting point, electron configuration",
        #             "host cities, number of participating countries, total number of events, medal tally, records broken",
        #             "categories , year of award, country of origin, field of contribution" ]

        types = ["Intercomparison",
                 "Superlative",
                 "Aggregation",
                 "Distribution Compliance",
                 "Correlation Analysis",
                 "Variance Analysis",
                 "Descriptive Relationship",
                 "Hypothetical Scenarios"]
        keyword = ["member of"]
        que_txt = ''
        count = 0
        qtype = ''
        qq = ''

        total_iterations = len(topics)
        progress_bar = tqdm(total=total_iterations)
        for i, topic in enumerate(topics):
            # print(topic)
            prope = properties[0]
            topicid = self.wikiutils.get_wikidata_id(topic)
            # property_id = search_wikidata_property(keyword)[0][0]
            property_id = 'P463'
            # print(topicid)
            # print(property_id)
            # property_id = "P54"  # Example: Wikidata property for "award received" P166
            # value_id = "Q18748039"  # Example: Wikidata item for "ACM Fellow"
            titles = self.wikiutils.get_entities_wikipedia_titles(property_id, topicid)
            # print(len(titles))
            entities = []

            for title in titles:  # Limiting to first 5 results for demonstration
                # print(title)
                first_paragraph = self.wikiutils.get_wikipedia_first_paragraph(title)
                # print(f"Title: {title}\nFirst Paragraph: {first_paragraph}\n")
                entities.append({'title': title, 'firstparagraph': first_paragraph})
            # print(entities)
            topic_i = {'topic': topic, 'edge': keyword[0],
                       'properties': prope, 'entity': entities}
            fout_entity.write(json.dumps(topic_i) + '\n')

            table_n = topic.replace(' ', '')
            gen_q_prompt = '''Questions types are {},
            topic is {}, properties is {},
            using the type, topics , properties to ask 30 questions. The questions concern about only the listed properties.
            The questions should be no hop, single hop, multi hops.
            Just output questions type， questions and SQL for each question in a format'Questions type:...\n Hops:...\n Questions:...\n SQL:...\n',
            For SQL, the table name is {}, the columns are the properties without ' '. no other word and symbol. remove the Serial number.'''.format(
                types, topic, prope, table_n)
            answer = self.GPT(gen_q_prompt, "GPT-4")
            answer = answer.replace('\n\n', '\n')
            for line in answer.split('\n'):
                # print(line)
                if "Questions type" in line:
                    qtype = line.split(':')[1]
                if "Questions" in line:
                    qq = line.split(':')[1]
                if "Hops:" in line:
                    qhops = line.split(':')[1]
                if "SQL" in line:
                    qsql = line.split(':')[1]
                    ques_i = {'qid': count, 'topic': topic, 'edge': keyword, 'class': 'multi-entities', 'hop': 'multi-hops',
                              'properties': prope, 'type': qtype, 'question': qq, 'sql': qsql, 'answer': 'NOT YET'}
                    count += 1
                    fout_queries.write(json.dumps(ques_i) + '\n')
            que_txt = que_txt + answer

            progress_bar.update(1)

        progress_bar.close()
        fout_queries.close()
        fout_entity.close()
        # print(answer)

        # for que in ques:

        # file_path = "Stran/queries.txt"

        # # Open the file in write mode and save the content
        # with open(file_path, "w") as file:
        #     file.write(que_txt)

    def gen_s_query(self, qfile='sing_entity_QA.jsonl', efile='entities_first_paragraph.jsonl'):

        fout_queries = open(os.path.join(self.path, qfile), 'a')
        # fout_entity = open(os.path.join('benchmarks_a/entities_first_paragraph.jsonl'), 'a')

        types = ["Intercomparison",
                 "Superlative",
                 "Descriptive Relationship",
                 "Hypothetical Scenarios"]

        count = 1

        # gen_q_prompt = '''Questions types are {},
        # Make 70 questions about single entity's property from wikipedia. The questions concern about only the listed types.
        # The questions should be no hop, single hop, multi hops.
        # Just output questions type， questions and answer for each question in a format'
        # Type:...\n  Hops:...\n Entity:...\n Question:...\n Ground Answer:...\n', no other word and symbol.
        # remove the Serial number.'''.format(types)

        with open(os.path.join(self.path, efile), 'r') as file:
            # Read each line in the file
            for line in file:
                # Convert the JSON string to a Python dictionary
                data = json.loads(line)

                # Process the data as needed
                # print(data)
                titles = data['entity']

                for entity in titles:  # Limiting to first 5 results for demonstration
                    # first_paragraph = get_wikipedia_first_paragraph(title)
                    # print(f"{name} (Wikidata ID: {wikidata_id}) is an ACM Fellow.")

                    title = entity['title']
                    first_paragraph = entity['firstparagraph']

                    gen_q_prompt = '''Make 12 questions about the {} of entity{} based on {}.Questions types are {}, the questions concern about only the listed types.
                                                            The questions should be no hop, single hop, multi hops. 
                                                            Just output questions type， questions and answer for each question in a format'Type:...\n  Hops:...\n Entity:...\n Question:...\n Ground Answer:...\n', 
                                                            no other word and symbol. remove the Serial number.'''.format(
                        data['properties'], title,
                        first_paragraph,
                        types)

                    answer = self.GPT(gen_q_prompt, "GPT-4")
                    # print(answer)
                    answer = answer.replace('\n\n', '\n')
                    for line in answer.split('\n'):
                        # print(line)
                        if "Type:" in line:
                            qtype = line.split(':')[1]
                        if "Question:" in line:
                            qq = line.split(':')[1]
                            # print(qq)
                        if "Hops:" in line:
                            qhops = line.split(':')[1]
                        if "Entity:" in line:
                            qen = line.split(':')[1]
                        if "Ground Answer:" in line:
                            qanswer = line.split(':')[1]
                            # print(qanswer)

                            ques_i = {'qid': count, 'topic': data['topic'], 'edge': data['edge'], 'class': 'sing entity',
                                      'Hops': qhops, 'Entity': qen, 'properties': data['properties'],
                                      'type': qtype, 'question': qq, 'answer': qanswer}

                            count += 1
                            fout_queries.write(json.dumps(ques_i) + '\n')
        fout_queries.close()

    def gen_answer(self, outfile='multi_entities_SQA.jsonl', qfile='SGraphQA_test.jsonl'):
        idx = 0
        fout_qa = open(os.path.join(self.path, outfile), 'a')
        with open(os.path.join(self.path, qfile), 'r') as file:
            # Read each line in the file
            for line in file:
                # Convert the JSON string to a Python dictionary

                data = json.loads(line)
                # print(data)
                if 'sql' in data:
                    sqll = data['sql'].split(';')[0]
                else:
                    sqll = ''
                table_n = data['topic'].replace(' ', '').replace('-', '')
                qtype = data['type']
                true_labl = "true"

                if ('Correlation Analysis' in qtype) or ('Variance Analysis' in qtype):
                    prompt = '''
                    The question is {}, Let's say we have a table that answers the question, just output the columns for the answer and related analysis method, no other words.
                    '''.format(data['question'])
                    answe = self.GPT(prompt, 'GPT-4')
                else:
                    # print(ques)
                    # sql_query = "SELECT * FROM {};".format(table_n)
                    answe = csv_sqltool(table_n, sqll)

                ques_i = {'qid': idx, 'topic': data['topic'], 'edge': data['edge'], 'class': data['class'],
                          'hop': data['hop'], 'properties': data['properties'], 'type': data['type'],
                          'question': data['question'], 'sql': sqll, 'goldanswer': answe,'true': true_labl}
                idx += 1
                fout_qa.write(json.dumps(ques_i) + '\n')

        fout_qa.close()
