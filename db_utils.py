# -*- encoding: utf-8 -*-

import pandas
import sqlite3
import pandas as pd

class dbutils(object):

    def __init__(self, **kwargs):
        self.db = 'sqlite3'

    def csv_sqltool(self, table, sql_query):
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
            print("Database error:", e)
            print("An error occurred:", e)
            return 'Wrong'
