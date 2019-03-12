# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:19:16 2019

@author: ASUS
"""

import tensorflow as tf
max_tokens = 544
pad = 'pre'

text = ''' 'Released Christmas Day in 2002 to IMAX and large format screens, The Lion King makes a triumphant return to the screen after eight years. Its every bit as majestic and great as it has been before.The Lion
 King Mufasa (James Earl Jones) just had a baby cub named Simba. All of the animals come to the ceremony, except for Mufasa\'s brother Scar (Jeremy Irons). Scar desperately wants to be King, but can\'t. As long
 as Mufasa and Simba are there. Soon Simba is able to walk and talk and is voiced by Jonathon Taylor Thomas. After hearing about an elephant graveyard from Scar, he and his friend Nala (Niketa Calame) visit it.
 They meet three bumbling hyenas: Banzai (Cheech Marin), Shenzi (Whoopi Goldberg), and Ed (Jim Cummings), but they manage to leave unhurt. Scar is upset that the hyenas didn\'t do the job, so he orders a stampe
de to wipe out both of them, but it only takes care of Mufasa. Scar convinces Simba that he killed Mufasa, not Scar. So Simba flees into exile.The Lion King really benefits from the larger screen. Its lavish la
ndscapes will be able to capture you more, and you can really savor the animation. Disney didn\'t need any humans, so they could spend all of the time on a great story and lush landscapes. In fact, its camera m
ovement was so majestic that you actually felt like you were part of the pride of lions.The music boomed and really created the atmosphere. Although I had seen this picture before, I still was tense because of
the way the music played out. Most of the time, I would just roll my eyes at the attempt to make me nervous. But Hans Zimmer\'s music really bowled me over and made my heart do calisthenics. Unlike such new Dis
ney pics like Lilo and Stitch, the songs actually did some good. They took you out of a somewhat dreary mood and put a smile on your face and made your feet want to tap along. There were only a few, but they we
re very entertaining. And the Circle of Life song at the beginning was beautiful, with its perfect pictures and perfect sound.I really like James Earl Jones (he\'s pretty diverse), and this time was no exceptio
n. He seemed to act like he didn\'t want to do this role, but he couldn\'t contain his excitement for wanting to do voice-overs again (he had done some work in The Simpsons before). Matthew Broderick redeemed h
imself for me (after the atrocious Ferris Bueller\'s Day Off) by showing a strong voicing as the adult Simba. Cheech Marin, Whoopi Goldberg, and Jim Cummings really had good chemistry together, even though they
 didn\'t have much screen time. Irons was really good and creepy as Scar (one of those who you can\'t help but hate), and if that is him really singing, brava!Be warned, The Lion King isn\'t really for youngste
rs. It had intense thematic elements that should have warranted a PG, instead of those that don\'t deserve it (Lilo and Stitch, again). The mood that the music and the script brought out could damper your day,
so be warned.This is one movie where you can feel for the characters. You don\'t say `haha, he\'s dead\', you say `Gasp! I\'m so sad!\' If it weren\'t for the gifted scriptwriters, this movie would be kaput and
 a nothing, not the best Disney movie ever made.The Lion King is a majestic movie, not without humor, that is for almost all to see.My rating: 9/10Rated G for intense thematic elements.' '''

texts = [text]

model = tf.keras.models.load_model("sentiment-CNN.model")

tokens = tokenizer.texts_to_sequences(texts)

type(tokens)

tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)


print(model.predict(tokens_pad))