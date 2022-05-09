# Disaster Response Pipeline Project

### Table of contents

1. [Project Description](#motivation)
2. [Installation](#installation)
3. [File Description](#file)
4. [Results](#results)
5. [Instructions](#instructions)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Description <a name="motivation"></a>

This project is the next project of the Udacity DataScience nanodegree program. Within this project one is setting up a 
data-pipeline and a machine-learning-pipeline. The goal is to classify message data sent from people during disasters to 
label and connect to help services.

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