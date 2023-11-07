# AutoLabeler

AutoLabeler is a project designed to automate the labeling process of datasets. This readme file will guide you through setting up the project, providing configuration details, and managing your labeled and unlabeled datasets.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Project Structure](#project-structure)
4. [Dataset Management](#dataset-management)
5. [Usage](#usage)

## 1. Introduction

AutoLabeler is a tool that helps streamline the data labeling process. It takes a seed dataset (seed.csv) with example labeled data and an unlabeled dataset (test.csv) and applies a labeling algorithm based on your configuration to automatically label the test dataset.

## 2. Getting Started

### Prerequisites

Before you start, ensure you have the following:

- Python 3.x
- pip
- evadb

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Preethi1609/evadb-auto-label
   pip install evadb
## 3. Project Structure
1. To add a new example, follow the following directory structure:
    ```
    new-example/
        |-- new_example_config.json
        |-- new_example_notebook.ipynb
        |-- data/
        |   |-- seed.csv
        |   |-- test.csv

3. The new_example_config.json would look something like this:
    ```
    {
        "task_name": "SpamClassification",
        "task_type": "classification",
        "dataset": {
        "label_column": "class",
        "label_separator": ", ",
        "delimiter": ","
        },
        "prompt": {
        "task_guidelines": "You are an expert at xxx. Your goal is to xxx. Your job is to correctly label the provided input example into one of the following categories:\n{labels}",
        "output_guidelines": "You will return the answer as a comma separated list of labels sorted in alphabetical order. For example: \"label1, label2, label3\"",
        "labels": [
            "spam",
            "ham"
        ],
        "few_shot_examples": "spam-ham-label/data/seed.csv",
        "example_template": "Input: {example}\nOutput: {labels}\n"
        }
    }
  


## 4. Dataset Management

In the data/ directory, you should place your datasets:

    seed.csv: This is your seed dataset with example labeled data.
    test.csv: This is the unlabeled dataset that AutoLabeler will process.

## 5. Usage
In the Jupyter Notebook script, you will:

1. Establish a connection to the "evadb".
2. Define a function called "AutoLabel," which is implemented in the "./functions/autolabel.py" module.
3. Create a table to store the test data and load this data into the "evadb".
4. Utilize a select query on the User-Defined Function (UDF) to execute the previously defined "AutoLabel" function and display the results.

