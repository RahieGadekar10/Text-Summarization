import torch
import flask
import time
from flask import Flask, render_template, request
from flask import request
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import torch.nn as nn


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
""" 
@app.route('/predict', methods=['POST','GET'])
def predict() : 
    name = "hello oeiple"
    result = request.form["sentence"]
    return render_template('index2.html', result=result )
 """
@app.route('/prediction', methods=['POST'])
def prediction() : 
    model = T5ForConditionalGeneration.from_pretrained("model")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    generator = pipeline("summarization" ,model=model, tokenizer=tokenizer)
    results = generator(str(request.form['text']))   
    result =  results[0]['summary_text']
    return render_template('index.html', result=result )

if __name__ == "__main__":
    app.run()

