import fastbook
from fastbook import *
from fastai.vision.widgets import *
from fastcore.foundation import L 
import gradio as gr

fastbook.setup_book()


learn = load_learner('export.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


title = "Liverpool FC Player Classifier"
description = "A Liverpool FC Player Classifier created using FastAI. "
interpretation='default'
enable_queue=True


iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(), 
    outputs=gr.Label(num_top_classes=3) 
    )

iface.launch(share=True)