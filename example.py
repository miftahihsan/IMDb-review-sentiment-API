import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# cPickle is for v2
# import cPickle
import pickle


class example():

    def predict(self, reviewList, data_text):
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
        # data_text = ""
        if data_text == "":
            with open("data_text_python2.pickle", "rb") as f:
                # cPickle is for v2
                # data_text = cPickle.load(f)
                data_text = pickle.load(f)
            
        tokenizer.fit_on_texts(data_text)

        tokens = tokenizer.texts_to_sequences(reviewList)

        tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

        model = tf.keras.models.load_model("jupyter_save_v1.h5")


        score_list = model.predict(tokens_pad)

        total_score = 0

        best_review_index = 0
        best_review_score = 0

        counter = 0
        for i in score_list:

            if i > best_review_score:
                best_review_score = i
                best_review_index = counter

            total_score += i
            counter = counter + 1

        
        
        return total_score / len(score_list), best_review_index

        # return model.predict(tokens_pad)

