---
title: Ham Or Spam
emoji: 🔥
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 4.14.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Ham or Spam Classifier with TensorFlow and Gradio

## Project Overview
This project presents a machine learning solution for classifying text messages into 'Ham' (non-spam) or 'Spam'. It uses TensorFlow to build a Recurrent Neural Network (RNN) and deploys a Gradio interface for easy interaction with the model. You can check it out over here! https://huggingface.co/spaces/Arsalan8/ham_or_spam

## Key Technologies
- Python
- TensorFlow
- Gradio
- Pandas

## Installation
To get started, ensure Python is installed on your system and then install the required libraries:
```bash
pip install pandas tensorflow gradio
```

## Data Preprocessing
The dataset, assumed to be named 'spam.csv', is preprocessed as follows:
- Reading the CSV file.
- Converting the class labels to binary format (0 for 'ham', 1 for 'spam').
- Splitting the data into training and testing sets.

## Model Architecture
The TensorFlow model includes:
- TextVectorization Layer: Processes the text data.
- Embedding Layer: Converts text to dense vector embeddings.
- Bidirectional LSTM Layer: Captures the context from both directions of text sequences.
- Dense Layers: For classification output.

## Model Training
- The model is compiled with Binary Crossentropy loss and Adam optimizer.
- It is trained for 15 epochs on the training dataset and validated on the test dataset.

## Model Evaluation
After training, the model is evaluated to determine its accuracy and loss:
```
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
```

## Deployment with Gradio
The trained model is deployed using Gradio, which provides a user-friendly interface for real-time predictions:
```
iface = gr.Interface(
    fn=predict_spam, 
    inputs="text", 
    outputs="text",
    title="Ham or Spam Classifier",
    description="A Ham or Spam Classifier created using TensorFlow. Input a message to see if it's classified as Ham or Spam!"
)
iface.launch(share=True)
```

## Running the Project
- Clone the project repository.
- Install the required dependencies.
- Execute the Python scripts for training the model and launching the Gradio interface.

## Conclusion
This project demonstrates the use of TensorFlow and Gradio to build and deploy a practical machine learning solution for text classification. The model effectively distinguishes between 'Ham' and 'Spam' messages, making it a useful tool for email filtering or similar applications.


