from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def tired():
    #some computations
    #load our model and get the output
    return render_template('index.html', prediction_text='yes'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
