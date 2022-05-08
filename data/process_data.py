import sys

import pandas as pd
from sqlalchemy import create_engine

class DataPipeline_DisasterResponse():

    """
    A DataPipeline object:
    - holds the necessary filepaths ,
    - reads in the data for this project,
    - transforms the data,
    - stores the data in a SQLite database


    Arguments:
        messages_filepath   -- the filepath to the file messages.csv
        categories_filepath -- the filepath to the file categories.csv
        database_filepath   -- the filepath where to store the resulting database

    Attributes:
        messages_filepath   -- the filepath to the file messages.csv
        categories_filepath -- the filepath to the file categories.csv
        database_filepath   -- the filepath where to store the resulting database
        df                  -- the dataframe which holds the data
    """

    def __init__(self, messages_filepath, categories_filepath, database_filepath):
        self.messages_filepath = messages_filepath
        self.categories_filepath = categories_filepath
        self.database_filepath = database_filepath
        self.df = pd.DataFrame()

    def load_data(self):
        """
        Given the filepaths of the two datasources the data is read in, merged and stored in the df attribute.
        """
        messages = pd.read_csv(self.messages_filepath)
        categories = pd.read_csv(self.categories_filepath)
        self.df = messages.merge(categories, on="id")


    def clean_data(self):
        """
        The cleaning of the data is done via the following procedure:

        1. Step:
        The 36 different categories are extracted from the categories column, and separated into 36 own columns.

        2. Step:
        Extraction of the numeric value out of the string of each datapoint.

        3. Step:
        Delete the old categories column and add the new categories value to the attribute self.df

        4. Step:
        If there are ducplicates, then remove those
        """
        # 1. Step
        categories = self.df.categories.str.split(pat=';', expand=True)
        row = categories[:1]
        category_colnames = [row[x][0][:-2] for x in row]
        categories.columns = category_colnames

        # 2. Step
        for column in categories:
            # each value to be the last character of the string
            categories[column] = categories[column].str[-1:]
            # convert column from string to numeric
            categories[column] = pd.to_numeric(categories[column])

        # 3. Step
        self.df = self.df.drop('categories', axis=1)
        self.df = pd.concat([self.df, categories], axis=1)

        # 4. Step
        # Elimination of duplicates
        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates()



    def save_data(self):
        """
        
        """
        pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()