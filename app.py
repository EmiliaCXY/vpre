from flask import Flask, request, jsonify, render_template


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #some computations
    #load our model and get the output
    # with open("ref.csv", 'r', newline='') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         output = row[0]
    #
    # count = 0
    # for i in output:
    #     if count % 40 == 0:
    #         front = output[:count]
    #         back = output[count:]
    #         output = front + "\n" + back
    #     count += 1
    output = ''

    for x in request.form['sequence']:
        output = output + x
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
