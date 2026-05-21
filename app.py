import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained LSTM model
model = load_model("next_word_lstm.h5")

# Load tokenizer
with open("tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)

# Function to predict next word
def predict_next_word(model, tokenizer, text):
    
    max_sequence_len = model.input_shape[1] + 1

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return "Word not found"


# Streamlit app
st.title("Next Word Prediction using LSTM ")

st.write("Enter a sentence and the model will predict the next word.")

input_text = st.text_input(
    "Enter your text",
    "To be or not to be"
)

if st.button("Predict Next Word"):

    next_word = predict_next_word(
        model,
        tokenizer,
        input_text
    )

    st.success(f"Next Word Prediction: {next_word}")