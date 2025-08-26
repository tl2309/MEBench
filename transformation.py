# -*- encoding: utf-8 -*-


from txtai import LLM
import sqlite3
import csv
from tqdm import tqdm
import os
import json

llm = LLM("/home/Mistral-7B-OpenOrca-AWQ")

def count_jsonl_lines(file_path):
    with open(file_path, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
        return line_count


class Transfer(object):

    def __init__(self, file='entities_first_paragraph.jsonl', **kwargs):
        self.file = file

    def trans(self, path):
        total_iterations = count_jsonl_lines(os.path.join(path, self.file))
        progress_bar = tqdm(total=total_iterations)
        with open(os.path.join(path, self.file), 'r') as file:
            # Read each line in the file
            for line in file:
                # Convert the JSON string to a Python dictionary
                data = json.loads(line)

                # Process the data as needed
                # print(data)
                topic = data['topic']#table
                titles = data['entity']
                properties = data['properties']

                context = ""
                conn = sqlite3.connect('document_store.db')
                cursor = conn.cursor()
                propertys = properties
                property_i = ['title']
                for line in propertys.split(','):
                    property_i.append(line.replace(' ', ''))

                topic = topic.replace(' ', '').replace('-', '')
                table_name = str(topic)
                cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
                columns_with_types = ', '.join([f'{col} TEXT' for col in property_i])
                print('columns_with_types= ', columns_with_types)
                cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                  {columns_with_types}
                )
                ''')

                # for node in list(g.scan()):
                #   gtext = g.attribute(node, "text")
                #   gname = g.attribute(node, "id")

                #   wikidata_id = get_wikidata_id(gname)
                # print(titles)
                for entity in titles:  # Limiting to first 5 results for demonstration
                    # first_paragraph = get_wikipedia_first_paragraph(title)
                    # print(f"{name} (Wikidata ID: {wikidata_id}) is an ACM Fellow.")

                    title = entity['title']
                    first_paragraph = entity['firstparagraph']
                    prompt = f"""<|im_start|>system
                    You are a friendly assistant. You answer questions from users.<|im_end|>
                    <|im_start|>user
                    Extracting property value using only the context below.If the property value is not mentioned in the context, using 'None'to fill.
                    The output size must meet the size of property. Just output the property value in order one by one.
                    property: {propertys}
                    context: {first_paragraph} <|im_end|>
                    <|im_start|>assistant
                    """
                    data = llm(prompt)
                    # print(data)
                    value_i = ['\'' + title + '\'']
                    for line in data.split('\n'):
                        if len(line.split(':')) > 1:
                            # print(line.split(':')[1])
                            value_i.append('\'' + line.split(':')[1].replace(',', '').replace('\'', '') + '\'')
                    # print(len(property_i))
                    if len(value_i) == len(property_i):
                        # print(len(value_i))
                        # print(len(property_i))
                        sql = f'INSERT INTO {table_name} ({", ".join(property_i)}) VALUES ({", ".join(value_i)})'
                        # print(sql)
                        cursor.execute(sql)
                        # context.join(data)

                cursor.execute(f'SELECT * FROM {table_name}')
                # Fetch all rows from the query result
                rows = cursor.fetchall()
                # Close the cursor and connection to the database
                cursor.close()
                conn.close()

                # Write the fetched data to a CSV file
                with open(f'outputcsv/{topic}.csv', 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    # Write the header row if needed
                    csv_writer.writerow([column[0] for column in cursor.description])
                    # Write the data rows
                    csv_writer.writerows(rows)

                print("CSV file has been created successfully.")

                progress_bar.update(1)

            progress_bar.close()
