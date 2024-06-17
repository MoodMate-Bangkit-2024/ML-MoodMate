# MoodMate-ML

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Library](#Library)
- [Model](#Model)
- [Evaluation](#evaluation)
- [Model Conversion](#)

## Introduction
This repository contains documentation for the Machine Learning component of the MoodMate Capstone Project. We are developing two main features for this project:

* **Mood Detection from Journaling**: This feature classifies user journal entries into one of four mood labels: anger, sadness, fear, and joy.
* **Chat Bot**: This feature enables interaction with users.
  
This document provides a comprehensive overview of the machine learning methodologies, models, and tools used in developing these features.

## Datasets
Our project leverages multiple datasets sourced from various repositories to support diverse aspects of our research and development. These datasets are integral for training and evaluating our machine learning models and algorithms.

The specific datasets used include:
* [MoodDetection](https://github.com/diapica/EmotionDetection/blob/29d486253662d435b61930654bdc224709efa86e/Dataset/Twitter_Emotion_Dataset.csv)
* [Emotion-Dataset-from-Indonesian-Public-Opinion](https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/tree/3368985e79b08309af6b13fcc75901036a107974/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion)
* [Data_MoodTracker](https://github.com/codewithriza/MoodTracker/blob/ed6078071d2a64ec7c54068fc318c9bab0215976/NLP-Text-Emotion/data/emotion_dataset_2.csv)
* [Emotions Data for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=train.txt)'

Additionally, we have merged several part of these datasets into a consolidated dataset available at [dataset](https://github.com/MoodMate-Bangkit-2024/ML-MoodMate/blob/fe898a39ac08109d7d01c92d894a299d2300840d/Model/DataDict/databagus.csv). This merged dataset contains entries categorized into four mood labels: Anger, Fear, Sadness, and Happiness. For training our models, we have utilized a dataset comprising 23.546 rows of data.

## Library

In the MoodMate-ML project, various libraries and tools are used for data processing, model building, and evaluation. Below is a brief explanation of each:

#### Data Manipulation and Analysis
- **Pandas** (`import pandas as pd`): For handling and processing datasets.
- **NumPy** (`import numpy as np`): For numerical computations and handling arrays.

#### Machine Learning and Deep Learning
- **TensorFlow** (`import tensorflow as tf`): For building, training, and deploying machine learning models.
  - **Tokenizer**: Converts text into sequences.
  - **pad_sequences**: Ensures all input sequences have the same length.
  - **Adam**: An optimizer for training models.
  - **l2**: A regularization method to prevent overfitting.

#### Text Processing and Natural Language Processing (NLP)
- **re** (`import re`): For text preprocessing using regular expressions.
- **emoji** (`import emoji`): For handling emojis in text.
- **nltk**: For various NLP tasks.
  - **word_tokenize**: Splits text into words.
  - **stopwords**: Removes common words that are not meaningful.
  - **CRFTagger**: Tags words with their part of speech.

#### Indonesian Language Processing
- **Sastrawi** (`from Sastrawi.Stemmer.StemmerFactory import StemmerFactory`): For stemming Indonesian words.

#### Visualization
- **Matplotlib** (`import matplotlib.pyplot as plt`): For creating visualizations of data and model performance.

#### Custom Dataset Handling
- **Excel**:
  - **pd.read_excel**: Reads data from Excel files.
- **CSV**:
  - **pd.read_csv**: Reads data from CSV files.

#### Model Conversion
- **TensorFlow.js Converter**: Converts Keras models to TensorFlow.js format for web deployment.
  - `!tensorflowjs_converter --input_format keras 1my_keras_model_fix.h5 tfjs_model`

These libraries are essential for the development and implementation of the MoodMate-ML project's features.


## Model

The MoodMate-ML project includes two primary features: Mood Detection from Journaling and a Chat Bot. Each feature employs specific machine learning models and architectures to achieve its functionality. Below are detailed explanations and links to the respective models.

#### Mood Detection from Journaling
This feature classifies user journal entries into one of four mood labels: anger, sadness, fear, and joy. The model architecture for mood detection includes preprocessing steps to clean and tokenize text data, followed by a neural network to classify the moods accurately. For a detailed explanation and implementation, please refer to the [Mood Detection Model](https://github.com/MoodMate-Bangkit-2024/ML-MoodMate/blob/0e2119c9ceee075cd02858428784ccc635607840/Model/TFJS%20MODLES/Capstone.ipynb).

#### Chat Bot
The Chat Bot feature enables interaction with users, providing responses based on user inputs. For a comprehensive overview of the chat bot model, please visit the [Chat Bot Model](https://github.com/MoodMate-Bangkit-2024/ML-MoodMate/blob/fe898a39ac08109d7d01c92d894a299d2300840d/Model%20Chatbot/Python%20Model/Model_ChatBot_(MoodMate).ipynb).

These models are integral to the MoodMate-ML project, leveraging advanced machine learning techniques to provide accurate mood detection and interactive user experiences.


## Evaluation

The MoodMate-ML project includes two primary features: Mood Detection from Journaling and a Chat Bot. Below is the evaluation of the Mood Detection from Journaling model based on its accuracy during training, validation, and testing. Additionally, key performance metrics are provided, along with a confusion matrix for the testing data.

### Mood Detection from Journaling
This feature classifies user journal entries into one of four mood labels: anger, sadness, fear, and joy.

#### Training and Validation Accuracy
The model was trained and validated using a dataset of 2,023,902 entries. Below are the accuracy metrics for training and validation:

- **Training Accuracy**: 90.4%
- **Validation Accuracy**: 78.71%

#### Testing Performance
The model was tested with a separate dataset to evaluate its performance. Below are the key performance metrics:

- **Accuracy**: 92.60%
- **Precision (Weighted)**: 92.86%
- **Recall (Weighted)**: 92.60%
- **F1 Score (Weighted)**: 92.34%

#### Confusion Matrix
The confusion matrix below provides a detailed view of the model's performance on the test data, showing the actual versus predicted mood labels.

![Confusion Matrix](https://github.com/MoodMate-Bangkit-2024/ML-MoodMate/blob/main/Model/lib/image.png?raw=true)

These models are integral to the MoodMate-ML project, leveraging advanced machine learning techniques to provide accurate mood detection and interactive user experiences.



## Model-Conversion

To deploy the Mood Detection model on the web, we convert the Keras model into a TensorFlow.js format. Below is a brief explanation of the conversion process and the steps involved.

### Conversion Steps

#### 1. Tokenize Text Data
Use the `Tokenizer` from `tensorflow.keras.preprocessing.text` to convert text into sequences and create a word index. The word index is saved as a JSON file for use during inference.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# Get the word index from the tokenizer
word_index = tokenizer.word_index

# Save word index to a JSON file
with open('1word_index12_fix.json', 'w') as f:
    json.dump(word_index, f)
```

#### 2. Save the Keras Model
Save the trained Keras model to an H5 file, which serves as the source model for conversion.

```python
model.save('1my_keras_model_fix.h5')
```

#### 3. Convert to TensorFlow.js Format
Use the tensorflowjs_converter to convert the Keras model to TensorFlow.js format, which includes generating model.json and binary weight files.

```python
!tensorflowjs_converter --input_format keras 1my_keras_model_fix.h5 tfjs_model
```

#### 4. Verify the Conversion
List the contents of the tfjs_model directory to ensure the conversion was successful and to obtain the necessary model files.
```python
import os
os.listdir('tfjs_model')
```
#### Output Files
* 1word_index12_fix.json: Contains the word index for tokenizing text data.
* 1my_keras_model_fix.h5: The original Keras model file.
* tfjs_model/: Directory containing the TensorFlow.js model files (model.json and binary weight files).
  
These steps ensure that the model and its metadata are properly converted and ready for deployment in a web environment, enabling the Mood Detection feature to be accessible online.

