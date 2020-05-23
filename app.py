from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend
import numpy as np
import os


app = Flask(__name__)
ALLOWED_EXTENSIONS = {'fasta'}

@app.route('/')
def home():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    UPLOAD_FOLDER = 'uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        f = request.files['real-file']
        filename = secure_filename(f.filename)
        if f and allowed_file(filename):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output = predict()
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('index.html', prediction_text=output)
    return render_template("index.html")


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


vocab = ['A', 'T', 'G', 'C']
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate (not including seed sequence)
  num_generate = 20                                       #!!!

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0                                           #!!!

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


def predict():

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model-weights.h5')
    batch_size = 1
    new_model = build_model(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024, batch_size=batch_size)
    weights = loaded_model.get_weights()
    new_model.set_weights(weights)
    new_model.build(tf.TensorShape([1, None]))
    output = generate_text(new_model, start_string=u"ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTTTCTAGTCAGTGTGTT")


    return output



if __name__ == "__main__":
    app.run(debug=True)
