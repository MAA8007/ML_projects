import tensorflow as tf
import gradio as gr
import tensorflow as tf
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



model = tf.keras.models.load_model('hatespeech/best_model')

def preprocess_text(text):
    # Lowercasing the text
    text = text.lower()
    
    # Removing special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization (convert text into tokens)
    tokens = text.split()
    
    # Re-join tokens
    text = ' '.join(tokens)
    
    return text


def predict_spam(message):
    processed_text = preprocess_text(message)
    pred_prob = model.predict([processed_text])[0][0]
    label = "Hate Speech" if pred_prob > 0.5 else "Not Hate Speech"
    confidence = f"{pred_prob * 100:.2f}%" if label == "Hate Speech" else f"{(1 - pred_prob) * 100:.2f}%"
    return f"{label} (Confidence: {confidence})"






iface = gr.Interface(
    fn=predict_spam, 
    inputs="text", 
    outputs="text",
    title="Hate Speech Detector",
    description="A Hate Speech detection tool created using TensorFlow. Input a message to check it out!",
)

iface.launch(share=True)
