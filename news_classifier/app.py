
import tensorflow as tf
import gradio as gr
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer

# Load model
model = tf.keras.models.load_model('news classifier/news_classifier_optimized')

label_binarizer = LabelBinarizer()
label_binarizer.fit(['U.S. NEWS', 'COMEDY', 'PARENTING', 'WORLD NEWS', 'CULTURE & ARTS', 'TECH',
 'SPORTS', 'ENTERTAINMENT', 'POLITICS', 'WEIRD NEWS', 'ENVIRONMENT',
 'EDUCATION', 'CRIME', 'SCIENCE', 'WELLNESS', 'BUSINESS', 'STYLE & BEAUTY',
 'FOOD & DRINK', 'MEDIA', 'QUEER VOICES', 'HOME & LIVING', 'WOMEN',
 'BLACK VOICES', 'TRAVEL', 'MONEY', 'RELIGION', 'LATINO VOICES', 'IMPACT',
 'WEDDINGS', 'COLLEGE', 'PARENTS', 'ARTS & CULTURE', 'STYLE', 'GREEN', 'TASTE',
 'HEALTHY LIVING', 'THE WORLDPOST', 'GOOD NEWS', 'WORLDPOST', 'FIFTY', 'ARTS',
 'DIVORCE'])


def predict_news(headline):
    pred_prob = model.predict([headline])[0]
    predicted_label = label_binarizer.classes_[pred_prob.argmax()]
    return f"Predicted Category: {predicted_label}"

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_news, 
    inputs="text", 
    outputs="text",
    title="News Classifier",
    description="A News Classifier created using TensorFlow. Input a headline and see the predicted category!",
)

iface.launch(share=True)
