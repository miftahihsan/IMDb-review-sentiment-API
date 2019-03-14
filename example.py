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

        lion_king = ['''
        This is a film that can entertain anyone young or old, I usually don't care for 
        animated movies but this film is the real deal, this is one of disney's best animated movies. 
        The animation is top notch and flawless. This film also features 
        superb work from the vocal cast James Earl Jones, Jeremy Irons, Whoopi Goldberg. This is a standout
        ''', '''
        The best Disney animated film ever...This film had it all, it was funny, emotional, had family drama,
        and above all, great animation and songs! My personal favorite character is Rafiki, the Baboon! I still 
        cant forget the line he says to Simbe, "Past can hurt, but as I see it, you can either run from it, or 
        learn from it!" it is so true! i loved Zazu's weirdness, and shenzi's humor, above all, i loved Pumba's 
        innocence and Timon's intelligence! In short, i found the film the best film ever... The voice cast is also
        great. Mathew did complete justice to Simba, and what can I say about Mufasa, He is the best King, and dad 
        anyone can ever get! The movie is not meant for kids, its meant to teach every adult a lesson...to find our 
        place in the great "Circle Of Life". I give this film a perfect 10.

        ''' , '''
        *Disclaimer: I only watched this movie as a conditional agreement. And I see films for free. I wouldn't be caught dead giving my hard earned money to these idiots.

        Well, to explain the depth of this 'film', I could write my shortest review, ever. Don't see this movie. It is by far the stupidest, lamest, most lazy, and unbelievably UNFUNNY movie I have ever seen. It is a total disaster. But since my hatred for this movie, and the others like it, extends far beyond one viewing, I think I'll go on for a bit.

        I don't know any of the people in the movie besides Carmen Electra, Vanessa Minnillo, and Kim Kardashian, but it doesn't matter. They're all horrible, though I think that was the point. The editing is flat out horrible, and possibly blatant continuity errors make this crapfast even crappier than I thought it would be. Now I know that these films are not supposed to be serious at all, but come on, it's film-making 101 that if someone gets a minor facial cut, it should be there in the next shot. AND, if someone gets cut by a sword, there should be blood and at least a cut (though since the Narnia films "get away with it", I'll give Disaster Movie a pass here).

        The 'jokes' are thoughtless and mindless physical gags that obviously take after some of the most popular movies of the last year (there's some from late 2007 as well, including 2 of our 5 Best Picture nominees).

        You know what the saddest thing about these stupid movies are? I don't care how much money they make, or how many cameos they have, these sorry ass excuses for films are taking away jobs from actors, writers, and directors that truly deserve the attention. Lionsgate, I thought you had better taste than this. You should be ashamed of yourselves for making this kind of crap. And as for Jason Friedberg and Aaron Seltzer? Burn in hell. You guys are contributing to the decline of western civilization. Correction...you are the CAUSE of the downfall of western civilization.
        ''']

        tokens = tokenizer.texts_to_sequences(lion_king)

        tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

        model = tf.keras.models.load_model("jupyter_save_v1.h5")


        return model.predict(tokens_pad)

