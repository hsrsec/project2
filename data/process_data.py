import sys

import pandas as pd
from sqlalchemy import create_engine

class DataPipeline():

    """
    
    """

    def __init__(self, messages_filepath, categories_filepath, database_filepath):
        """
        
        Keyword arguments:
        messages_filepath   -- the filepath to the file messages.csv
        categories_filepath -- the filepath to the file categories.csv
        database_filepath   -- the filepath where to store the resulting database
        """
        self.messages_filepath = messages_filepath
        self.categories_filepath = categories_filepath
        self.database_filepath = database_filepath
        self.df = pd.DataFrame()

    def load_data(self):
        """
        
        """
        messages = pd.read_csv(self.messages_filepath)
        categories = pd.read_csv(self.categories_filepath)
        self.df = messages.merge(categories, on="id")


    def clean_data(self):
        pass


    def save_data(self):
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