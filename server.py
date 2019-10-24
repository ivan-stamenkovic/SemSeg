from flask import Flask, send_file, render_template, request

import time
import os
import sys

import model_runner
from server_inference import InferenceManager

inference = InferenceManager()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if (inference.initialized == False):
        inference.initialize()
    if 'inputFile' not in request.files:
        return 'No file selected.'
    file = request.files['inputFile']
    if file.filename == '':
        return 'No file selected.'
    file_dst = inference.process_image(file)
    return send_file(file_dst, mimetype="image/gif")

@app.route('/cleanup')
def cleanup():
    inference.deinitialize()
    return 'Cleanup success!'

if __name__ == "__main__":
    app.run(debug=True)
