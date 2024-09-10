# Email Spam Detection

This project is a spam detection system that uses Deep Learning techniques and a Long Short-Term Memory (LSTM) model for binary classification of emails as spam or not spam. The model is trained on a dataset of email messages and is designed to help identify unwanted emails.

## Project Overview

The project utilizes a dataset containing email messages, where each email is labeled as spam (`1`) or not spam (`0`). The goal is to preprocess the email text and train a machine learning model to classify new emails based on their content.

### Key Features

- **Text Preprocessing**: The text is cleaned by removing stopwords, punctuation, and tokenizing the emails.
- **Word2Vec**: The model uses word embeddings to convert words into dense vectors of fixed size.
- **LSTM Model**: A Sequential LSTM model is implemented to process the sequence of words in each email.
- **Early Stopping**: Prevents overfitting by stopping training when validation performance stops improving.

## Project Structure

- `NoSpam.ipynb`: The Jupyter notebook containing all the steps for data preprocessing, model building, training, and evaluation.
- `dataset/emails.csv`: The dataset used for training and testing the model. It contains two columns:
  - `text`: The email content.
  - `spam`: Label indicating whether the email is spam (`1`) or not spam (`0`).

## Installation

1. Clone this repository.
2. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the `dataset/` folder.

## Usage

1. Open the Jupyter notebook `NoSpam.ipynb`.
2. Run the notebook to preprocess the data, train the model, and evaluate its performance.

## Model Summary

The model uses an LSTM layer for sequence processing, with the following structure:

- **Input Layer**: Tokenized and padded email sequences.
- **Embedding Layer**: Converts words into dense vectors.
- **LSTM Layer**: Processes the sequence of word embeddings.
- **Dense Layer**: Fully connected layer for output classification (spam or not spam).

## Results

The model is trained on a dataset of email messages and evaluated based on accuracy, precision, recall, and F1-score.

## Acknowledgements

- Dataset: A publicly available email spam dataset.
- Libraries used: TensorFlow, Keras, NLTK, Pandas, Matplotlib, Seaborn.

