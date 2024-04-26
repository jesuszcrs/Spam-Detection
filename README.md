# Spam-Detection

# ML Classification (Spam Detection) Script README:
## Overview:
This script implements a machine learning model for classifying emails as spam or non-spam (ham) based on their content. It uses the Naive Bayes algorithm for classification and utilizes the scikit-learn library for data preprocessing, model training, and evaluation.

## Usage:
Ensure you have Python installed on your system.
Install the required libraries using pip install pandas scikit-learn.
Prepare your dataset in CSV format with 'text' and 'label' columns, where 'text' contains the email content and 'label' contains the corresponding class labels (spam or ham).
Update the script with the correct path to your dataset (spam_dataset.csv).
Run the script. It will preprocess the data, train the model, and evaluate its performance on a held-out test set.
The script will output the accuracy and classification report, showing precision, recall, F1-score, and other metrics.

## Dependencies:
Python 3.x
pandas
scikit-learn

## Dataset:
You can find example datasets for spam classification online, such as the SpamAssassin Public Corpus or the Enron Email Dataset.
