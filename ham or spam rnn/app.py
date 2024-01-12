import tensorflow as tf
import gradio as gr


model = tf.keras.models.load_model('ham or spam rnn/model')

def predict_spam(message):
    pred_prob = model.predict([message])[0][0]
    label = "Spam" if pred_prob > 0.5 else "Ham"
    confidence = f"{pred_prob * 100:.2f}%" if label == "Spam" else f"{(1 - pred_prob) * 100:.2f}%"
    return f"{label} ({confidence})"


iface = gr.Interface(
    fn=predict_spam, 
    inputs="text", 
    outputs="text",
    title="Ham or Spam Classifier",
    description="A Ham or Spam Classifier created using TensorFlow. Input a message to see if it's classified as Ham or Spam!",
)

iface.launch(share=True)
