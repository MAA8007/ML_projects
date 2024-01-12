# Liverpool FC Player Classifier

## Overview
The Liverpool FC Player Classifier is a deep learning application designed to classify images of Liverpool FC players. This project leverages the FastAI library, a high-level API on top of PyTorch, for efficient and accessible deep learning. It also uses Gradio for an easy-to-use web interface.

## Features
- **Data Collection**: Automates the collection of player images using the SerpApi.
- **Image Classification**: Employs a neural network using the ResNet34 architecture to classify player images.
- **Interactive Web Interface**: Offers a Gradio interface for easy interaction with the model.

## Requirements
- FastAI
- Requests
- Pandas
- PyTorch
- Gradio
- PIL (Python Imaging Library)

## Installation
To set up the Liverpool FC Player Classifier on your system, follow these steps:

1. Clone the repository:
```
git clone git@github.com:MAA8007/ML_projects.git
```

2. Navigate to the cloned directory:
```
cd liverpool_classifier
```

3. Install the required packages:
```pip install -r requirements.txt```


## Usage
To use the Liverpool FC Player Classifier, follow these instructions:

1. Execute the script to collect and download images:
```python app.py```

2. Access the web interface, upload an image, and see the classifier's predictions.

## Model Training
The model is trained on a dataset of images of various Liverpool FC players. It uses transfer learning with the ResNet34 architecture and is fine-tuned for specific player recognition. The classifier achieves a high accuracy of 93%, demonstrating its effectiveness in correctly identifying the players.

## Exporting and Loading the Model
- The trained model is exported to a `.pkl` file for easy loading and inference.
- The model can be loaded using FastAI's `load_learner` function for use in the Gradio interface.

## Gradio Web Interface
A Gradio web interface is provided for easy testing of the model. Users can upload an image of a Liverpool FC player, and the model will predict the player's name along with confidence scores.

## Contributing
Contributions to the Liverpool FC Player Classifier are welcome! Please refer to the contributing guidelines for more information on how to contribute.

## Acknowledgments
- FastAI for the deep learning framework.
- Gradio for the interactive web interface.
- SerpApi for the image collection functionality.


