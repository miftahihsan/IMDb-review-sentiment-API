import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import cPickle


class example():

    def predict(self, reviewList):
        pad = 'pre'
        num_words = 10000
        max_tokens = 544

        def tokens_to_string(tokens):
            # Map from tokens back to words.
            words = [inverse_map[token] for token in tokens if token != 0]
            
            # Concatenate all words.
            text = " ".join(words)

            return text

        tokenizer = Tokenizer(num_words=num_words)
        data_text = ""
        with open("data_text_python2.pickle", "rb") as f:
            data_text = cPickle.load(f)
            
        tokenizer.fit_on_texts(data_text)

        tokens = tokenizer.texts_to_sequences(reviewList)

        tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

        model = tf.keras.models.load_model("jupyter_save_v1.h5")


        score_list = model.predict(tokens_pad)

        total_score = 0

        for i in score_list:
            total_score += i
        
        return total_score / len(score_list)

        # return model.predict(tokens_pad)

