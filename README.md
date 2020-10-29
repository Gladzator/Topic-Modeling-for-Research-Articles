# Topic-Modeling-for-Research-Articles-FastAPI

## Table of Content
  * [Problem Statement](#problem-statement)
  * [Data](#data)
  * [Installation](#installation)
  * [Directory Tree](#directory-tree)
  * [Approach](#approach)
  * [Technologies/Libraries Used](#technologieslibraries-used)
  * [Run](#run)
  * [Screenshots](#screenshots)
  * [Team](#team)

## Problem Statement
Researchers have access to large online archives of scientific articles. As a consequence, finding relevant articles has become more difficult. Tagging or topic modelling provides a way to give token of identification to research articles which facilitates recommendation and search process. 

The research article abstracts are sourced from the following 6 topics: 
  - Computer Science
  - Mathematics
  - Physics
  - Statistics
  - Quantitative Biology
  - Quantitative Finance
  
 I have also built a web interface and API using FastAPI.

## Data
The dataset can be found [here](https://datahack.analyticsvidhya.com/contest/janatahack-independence-day-2020-ml-hackathon).
| Sr No.        | Column Name   | Description  |
| ------------- |:-------------:| -----:|
| 1 | ID | Unique ID for each article |
| 2 | TITLE | Title of the research article |
| 3 | ABSTRACT | Date when the pet arrived to the shelter |
| 4 | Computer Science | Whether article belongs to topic computer science (1/0)|
| 5 | Physics | Whether article belongs to topic physics (1/0) |
| 6 | Mathematics | Whether article belongs to topic Mathematics (1/0) |
| 7 | Statistics | Whether article belongs to topic Statistics (1/0) |
| 8 | Quantitative Biology | Whether article belongs to topic Quantitative Biology (1/0) |
| 9 | Quantitative Finance | Whether article belongs to topic Quantitative Finance(1/0) |

## Installation
The Code is written in Python 3.7.6. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install pipenv
pipenv install --dev
pip install python-multipart
```

## Directory Tree
```bash
├───.vscode
├───article_classification
│   ├───classifier
│   │   ├───__pycache__
│   │   ├───article_classifier.py
│   │   ├───model.py
├───api.py
├───assets
├───bin
├───env
├───model
|   ├───model.ipynb   
├───static
│   └───bootstrap
│       ├───css
│       └───js
├───templates
├───.flake8
├───.isort.cfg
├───config.json
└───Pipfile
```

## Approach
The two major parts of the project are:
  1. Model Building.
  2. Building an API which can be used to serve predictions over the browser.

The very first approach to modal building I took was to train the data with the use of BERT modal. The results were alright with a micro F1 score of ~ 81%. A pre-trained model which is more suited to the task at hand would most definitely give better results. Since the data at hand was of scientific articles, SciBERT pretrained modal was going to be a better choice. I combined the title and abstract field by separating them with '[SEP]' tokena and managed to achieve a micro F1 score of ~85%.

More about SciBERT can be found on this huggingface [link](https://huggingface.co/allenai/scibert_scivocab_uncased). If you wish to see my implementation of SciBERT using pytorch, you can find the same inside "modal/modal.ipynb".

I later served this model using FastAPI. To learn more about FastAPI visit [here](https://fastapi.tiangolo.com/).

## Run
Go to the root directory and type the following command in Terminal/Command Prompt
```bash
uvicorn article_classification.api:app
```


## Screenshots
<img target="_blank" src="https://user-images.githubusercontent.com/40065133/97539408-eaa8c180-19e7-11eb-97b1-a93d8167abe1.JPG" width="45%"><img target="_blank" src="https://user-images.githubusercontent.com/40065133/97541512-4de82300-19eb-11eb-850f-00cfa3d86be2.JPG" width="45%">


## Technologies/Libraries Used
![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://venturebeat.com/wp-content/uploads/2019/06/pytorch-e1576624094357.jpg?w=1200&strip=all" width=100>](https://pytorch.org/)[<img target="_blank" src="https://huggingface.co/front/assets/huggingface_logo.svg" width=100>](https://huggingface.co/)[<img target="_blank" src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" width=100>](https://fastapi.tiangolo.com/)


## Team
<img src="https://avatars2.githubusercontent.com/u/40065133?s=460&v=4" width="200" height="200">|
-|
Yash Vora

