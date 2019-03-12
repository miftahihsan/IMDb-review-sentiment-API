import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

class example():

    def predict(self):
        max_tokens = 544

        pad = 'pre'

        text = ''' Worst movie ever '''

        texts = [text]

        model = tf.keras.models.load_model("sentiment-CNN.model")

        number_of_words = 10000 # MEAN AMOUT

        # THIS IS WHERE THE DICTINARY IS MADE I.E. TOKENIZER.WORD_INDEX
        tokenizer = Tokenizer(number_of_words)

        tokens = tokenizer.texts_to_sequences(texts)

        tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                                padding=pad, truncating=pad)


        return model.predict(tokens_pad)