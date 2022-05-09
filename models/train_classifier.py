import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sys
import joblib
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet', 'omw-1.4'])

class MLPipeline_DisasterResponse():
    """
    A MLPipeline object:
    - has the necessary filepaths ,
    - creates the machine learning model and is fitting this model to the data
    - evaluates the performance of the best found model via cross-validation and grid-search
    - saves the model for further usesage


    Arguments:
        database_filepath   -- the filepath to the database
        model_filepath      -- the filepath where to store the resulting model

    Attributes:
        database_filepath   -- the filepath to the database
        model_filepath      -- the filepath where to store the resulting model
        X_train             -- part of the feature data to train the model
        X_test              -- part of the feature data to test the model
        Y_train             -- part of the response to train the model
        Y_test              -- part of the response to test and evaluate the model
        category_names      -- the category names
        model               -- the machine learning model
    """
    

    def __init__(self, database_filepath, model_filepath):
        self.database_filepath = database_filepath
        self.model_filepath = model_filepath
        self.X_train = []
        self.X_test = []
        self.Y_train = [] 
        self.Y_test = [] 
        self.category_names = []
        self.model = []

    def load_data(self):
        """
        The data is loaded and splitted into train and test sets 
        """
        engine = create_engine('sqlite:///' + self.database_filepath)
        df = pd.read_sql_table('DisasterResponse', con=engine)
        X = df['message']
        Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
        self.category_names = list(df.columns[4:])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, train_size=0.7)

    def tokenize(self, text):
        """
        Cleaning and tokenizing the text.
        """
        text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)
        return clean_tokens

    def build_train_model(self):
        """
        The machine learning pipeline is constructed and fitted to the train data. The GridSearch will be done with 
        defined parameter list.
        """
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
        ])
        parameters = {
            'vect__max_df': (0.75, 1.0),
            'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [100, 150],
            'clf__estimator__min_samples_split': [2, 3]
        }
        self.model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
        self.model.fit(self.X_train, self.Y_train)

    def evaluate_model(self):
        """
        Evaluate the fitted model and report the statistics to standard output
        """
        Y_pred = self.model.predict(self.X_test)
        for ind, column in enumerate(self.Y_test.columns):
            print(str("Column: " + str(ind) + " category: " + column + "\n"));
            print(classification_report(self.Y_test[column], Y_pred[:,ind], labels=np.unique(self.Y_test[column])));
            print("\n*************************************\n\n");


    def save_model(self):
        """
        Save the model at given (attribute) model_filepath
        """
        joblib.dump(self.model, self.model_filepath ,compress=1)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Creating MLPipeline Object...\n    DATABASE: {}'.format(database_filepath))
        mlPipeline = MLPipeline_DisasterResponse(database_filepath=database_filepath, model_filepath=model_filepath)
        
        print('Loading Data...')
        mlPipeline.load_data()

        print('Building and training the model...\nThis can last a few minutes...')
        mlPipeline.build_train_model()

        print('Evaluating model...')
        mlPipeline.evaluate_model()

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        mlPipeline.save_model()

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()