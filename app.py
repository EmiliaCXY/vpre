from flask import Flask, request, jsonify, render_template, url_for
from flask import send_from_directory
import os
import numpy as np



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    output = 'A'
    # output = generate_text(new_model, start_string=u"ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTTTCTAGTCAGTGTGTT")

    return render_template('index.html', prediction_text=output)



if __name__ == "__main__":
    app.run(debug=True)
