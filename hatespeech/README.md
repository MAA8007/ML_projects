---
title: Hatespeech Detector
emoji: 👀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.14.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Hate Speech Detection with TensorFlow and Gradio

## Project Overview

This project focuses on detecting hate speech in text data, particularly from tweets. It combines data preprocessing, TensorFlow-based model development, and deployment using Gradio for real-time, interactive predictions.

## Key Features

- **Deep Learning Model**: Built using TensorFlow for accurate text classification.
- **NLP Techniques**: Incorporates text preprocessing with NLTK, including tokenization, lemmatization, and stopword removal.
- **Interactive Interface**: Features a Gradio web interface for immediate model interaction and testing.

## Technologies

- Python
- TensorFlow
- Gradio
- NLTK
- Pandas

## Installation

First, ensure Python is installed. Then, install the required packages:

```bash 
pip install pandas tensorflow gradio nltk
```

## Model Architecture
The model includes:

- TextVectorization: Converts text to a numerical representation.
- Embedding: Translates tokens into dense vectors.
- Bidirectional LSTM: Captures context from both sequence directions.
- Dense and Dropout Layers: Adds non-linearity and prevents overfitting.

## Model Training
Training specifics:

- Loss function: Sparse Categorical Crossentropy.
- Optimizer: Adam with learning rate scheduling.
- Class weights for balancing.
- ModelCheckpoint for best model iteration.

## Model Evaluation
Post-training, the model is assessed on a test dataset:
```
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
```

## Deployment
Deployed using Gradio with a simple text input and output interface:


```
iface = gr.Interface(fn=predict_spam, inputs="text", outputs="text", ...)
iface.launch(share=True)
```

## Running the Project
- Clone the repository.
- Install all dependencies.
- Run scripts for preprocessing, training, and deploying the Gradio app.

## Conclusion
This project demonstrates TensorFlow and Gradio's combined power for building and deploying a machine learning model to detect hate speech. It is an effective tool for moderating online content.



