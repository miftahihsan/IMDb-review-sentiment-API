
# import numpy as np
# import keras as ks
# import tensorflow as tf
# import numpy as np
# from scipy.spatial.distance import cdist

# # from tf.keras.models import Sequential  # This does not work!
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, GRU, Embedding
# from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# import tensorflow as tf
# from tensorflow.python.keras.preprocessing.text import Tokenizer

from example import example

from flask import Flask


app = Flask(__name__)

@app.route('/')
def mainPage():

    return "<h1> <i> THIS IS A RESTRICTED AREA. <br> <br> YOU ARE NOT WELCOMED HERE!!! </i> </h1>"

@app.route('/predict')
def predict():

    ex = example()

    return str(ex.predict())
    

if __name__ == '__main__':
    app.run(debug=True)