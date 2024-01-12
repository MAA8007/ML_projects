---
title: NewsClassifier
emoji: 📚
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 4.14.0
app_file: app.py
pinned: false
---

# News Classifier

## Overview
The News Classifier is a machine learning application designed to classify news headlines into various categories. Utilizing TensorFlow and Gradio, this classifier offers an intuitive way to predict the category of a news headline based on a trained model. The application is hosted on Hugging Face Spaces and can be accessed via the following link: [News Classifier on Hugging Face Spaces](https://huggingface.co/spaces/Arsalan8/NewsClassifier).

## Features
- **Data Processing**: Ingests data from a JSON file, processes it using Pandas, and prepares it for model training.
- **Machine Learning Model**: Employs a TensorFlow model with a custom Attention Layer, Bidirectional LSTM, and several Dense layers for effective classification.
- **Gradio Interface**: Provides an interactive web interface for easy usage of the model.
- **Categories**: Capable of classifying news into multiple categories including U.S. News, World News, Comedy, Parenting, and more.

## Requirements
- TensorFlow
- Pandas
- Scikit-Learn
- Gradio
- Keras (for backend operations)

## Installation
To set up the News Classifier on your system, follow these steps:

1. Clone the repository:
```git clone git@github.com:MAA8007/ML_projects.git```

2. Navigate to the cloned directory:
```
cd news_classifier
pip install -r requirements.txt
```

## Usage
To use the News Classifier, you can either run it locally following the steps below or access it directly on Hugging Face Spaces:

- **Local Usage**:
1. Run the application:
  ```
  cd news_classifier
  python app.py
  ```
2. Open the provided Gradio interface link in your web browser.
3. Enter a news headline in the input field.
4. View the predicted category for the headline.

- **Hugging Face Spaces**:
- Visit the [News Classifier on Hugging Face Spaces](https://huggingface.co/spaces/Arsalan8/NewsClassifier) and follow the on-screen instructions to classify news headlines.

## Model Training
The model is trained on a dataset containing various news headlines and categories. It uses a combination of text vectorization, bidirectional LSTM, and a custom attention layer to effectively learn from the dataset.

## Model Saving and Loading
- The trained model is saved under `news_classifier_optimized`.
- It can be loaded using TensorFlow's `load_model` function for further use or modification.

## Contributing
Contributions to the News Classifier are welcome! Please refer to the contributing guidelines for detailed instructions on how to contribute.

## License
This project is licensed under the [LICENSE] - see the LICENSE file for details.

## Acknowledgments
- Dataset Source: [News Category Dataset v3]
- TensorFlow and Keras for the machine learning framework
- Gradio for creating interactive web interfaces

