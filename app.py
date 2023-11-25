import torch
import gradio as gr
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification

checkpoint = 'bert-base-uncased'
config = AutoConfig.from_pretrained('./config.json')
model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path='./model.safetensors', 
    config=config, 
    use_safetensors=True
)
classlabels = ['Negative', 'Positive']
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenizer_fn(text):
    return tokenizer(
        text, 
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

def inference_fn(text):
    model.eval()
    with torch.no_grad():
        output = model(**tokenizer_fn(text))
    prediction = torch.argmax(output.logits, dim=-1).item()
    return classlabels[prediction]


with gr.Blocks() as demo:
    gr.Markdown("<p style='text-align:center; font-size:24px; font-weight:bold;'>Sentiment Classification App</p>")
    gr.Markdown("<p style='text-align:center; font-size:15px;'>This is a super basic sentiment classification app that only predicts if sentiment is Positive or Negative.</p>")
    gr.Interface(
        fn=inference_fn,
        inputs=gr.Textbox(label="Input text", placeholder='Enter a random comment/review (e.g. The weather is lovely today.)'),
        outputs=gr.Textbox(label="Predicted sentiment"),
        allow_flagging='never'
    )

demo.launch()






