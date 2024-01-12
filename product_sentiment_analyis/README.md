# Real-World Sentiment Analysis: Flipkart Product Reviews

## Project Overview
This sentiment analysis project utilizes DistilBERT, a cutting-edge transformer model, for analyzing user-generated product reviews from Flipkart. It showcases how businesses can apply Natural Language Processing (NLP) techniques to derive valuable insights from customer feedback.

## Real-World Applicability
- **Business Insight:** Provides businesses with crucial insights by analyzing customer reviews.
- **User Reviews Training:** The model is trained on genuine user reviews, ensuring its effectiveness in processing and interpreting real customer opinions.

## Installation & Setup
To set up the project, install the necessary libraries using:
```bash
pip install datasets transformers evaluate
```

## Data Collection and Preprocessing
The foundation of this project lies in the sentiment analysis of authentic user reviews from Flipkart. The dataset was prepared through the following steps:
- **Data Collection:** Sourced from Flipkart's customer reviews.
- **Preprocessing Focus:** Concentrated on 'Summary' and 'Sentiment' columns.
- **Data Cleaning:** Ensured the cleanliness of the dataset for consistency and quality.
- **Label Conversion:** Converted sentiment labels into a numerical format for model training.

## Model Training
The project uses the DistilBERT model for sentiment analysis through transfer learning.
- **Model:** DistilBERT (Distilled Version of BERT)
- **Learning Rate:** 2e-5
- **Batch Size:** 16
- **Epochs:** 3
- **Evaluation Strategy:** At the end of each epoch

## Training Outcome
The model achieved around 95% accuracy, validating its efficiency in sentiment classification.

## Results
- **Training Loss:** 0.1373
- **Validation Accuracy:** Approximately 95%

## Deployment and Usage
The trained model is available on the Hugging Face Hub for easy access and deployment.
- **Model URL:** [Arsalan8/my_multiclass_model](https://huggingface.co/Arsalan8/my_multiclass_model)

## Sample Code for Model Usage
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Arsalan8/my_multiclass_model")
model = AutoModelForSequenceClassification.from_pretrained("Arsalan8/my_multiclass_model")

# Example text
text = "This product is great!"
inputs = tokenizer(text, return_tensors="pt")

# Prediction
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
predicted_label = model.config.id2label[predicted_class_id]

print(f"Predicted sentiment: {predicted_label}")
```

##Conclusion
This project successfully demonstrates the use of advanced NLP techniques in a real-world business context. The model's adaptability and accuracy make it an invaluable asset for comprehending and utilizing customer feedback effectively.


