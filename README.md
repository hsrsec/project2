# Disaster Response Pipeline Project

### Table of contents

1. [Project Overview](#summary)
2. [Project Description](#motivation)
3. [Installation](#installation)
4. [File Description](#file)
5. [Results](#results)
6. [Instructions](#instructions)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="summary></a>

FigureEight provided tweets and text messages from real disasters. The project is set up to find a way to filter these messages to identify
the most important ones. There is also a need to categorise these messages to be able to connect the right emergency service to the
needed help, i. e. when people need water to survive to send specific help.

## Project Description <a name="motivation"></a>

For providing a solution for this project FigureEight provided prelabeled tweets and text messages. This data is then transformed and prepared with an ETL Pipeline and the result is stored in a SQLite database. Afterwards a Machine Learning Pipeline is used to build a supervised learning model. Via the web app one can then copy the message under investigation into a web form and get a classification result, i. e.
the categories the algorithm identifies within the message. Afterwards a specific emergency response can then be activated.

## Installation <a name="installation"></a>

The code should run with no issues using Python version 3.*. You should have installed the packages you can find in the file
`requirements.txt`

## File Description <a name="file"></a>

There are four files:

1. LICENSE
- This project is licensed under the MIT License

2. README.md
- The file you are reading right now

3. requirements.txt
- The python packages used in this project

4. run.py
- The main application which runs the flask application


There are three folders:

1. data
- DisasterResponse.db       :   The SQLite database
- disaster_categories.csv   :   The dataset with categories
- disaster_messages.csv     :   The dataset with messages
- process_data.py           :   The data-pipeline class which reads, cleans and saves the data to the SQLite database

2. models
- train_classifier.py       :   The machine-learning-pipeline class, which trains and saves the model.

The Output-file `model.pkl` is missing here due to data capacity restrictions for large files from github. Just follow
the Instructions to run the file `train_classifier.py` to produce the file.

3. templates
- consists of html files for the flask application


## Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`

2. Run the following command in the main directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

This project is licensed under the MIT License. The data is provided by FigureEight and the script skeletons from Udacity.