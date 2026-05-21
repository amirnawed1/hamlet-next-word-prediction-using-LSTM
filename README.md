# hamlet-next-word-prediction-using-LSTM
LSTM-based NLP project for predicting the next word in a sentence using Shakespeare Hamlet text dataset with TensorFlow, Keras, and Streamlit.

# Next Word Prediction using LSTM

This project is a Deep Learning and NLP application that predicts the next word in a sentence using an LSTM (Long Short-Term Memory) neural network trained on Shakespeare's Hamlet dataset.

## Project Overview

The model learns language patterns from text data and predicts the most probable next word based on the input sentence.

Example:
Input: To be or not to
Prediction: be




## Technologies Used

- Python
- TensorFlow / Keras
- LSTM
- NLP
- NLTK
- Streamlit

---

## Features

- Text preprocessing using Tokenizer
- Sequence generation for training
- LSTM-based next word prediction
- Streamlit web application
- Early Stopping to prevent overfitting

---

## Project Structure

├── app.py
├── experiments.py
├── requirements.txt
├── tokenizer.pickle
├── next_word_lstm.h5
├── hamlet.txt
└── README.md


## Installation

Install all required libraries:
pip install -r requirements.txt


## Run the Streamlit Application

Start the application using:
bash-streamlit run app.py
After running the command, the app will open automatically in your browser.



## Run the Application
streamlit run app.py



## Model Training

The model was trained on Shakespeare Hamlet text data using:

- Embedding Layer
- LSTM Layer
- Dense Output Layer

Loss Function:categorical_crossentropy
Optimizer:Adam


---

## Future Improvements

- Add multi-word text generation
- Train on larger datasets
- Deploy on Streamlit Cloud
- Improve prediction accuracy

---

## Author

Amir Nawed
