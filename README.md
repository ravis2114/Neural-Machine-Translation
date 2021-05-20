# Neural-Machine-Translation
Project translates english language to hindi language in devanagari script. same script can be used for translating any one language to subsequent another language.

### Main model architecture to be trained
----
![Main Model to be trained](https://github.com/ravis2114/Neural-Machine-Translation/blob/main/model.png)

### Encoder Model generated from saved Model
----
![Encoder Model generated from saved Model](https://github.com/ravis2114/Neural-Machine-Translation/blob/main/encoder_model.png)

### Decoder Model generated from saved Model
----
![Decoder Model generated from saved Model](https://github.com/ravis2114/Neural-Machine-Translation/blob/main/decoder_model.png)


To translate using encoder-decoder model :
----
```
def translate(eng_sent):
  _, eh, ec = inf_enc_model.predict(eng_sent.reshape(1, len(eng_sent)))
  translated = []
  dec_inp_seq = np.array([2]).reshape(1,1) #2 for start token and 1 for end
  stop = False
  while not stop:
    d, [eh, ec] = dec_model_final.predict([dec_inp_seq, [eh, ec]])
    dec_inp_seq[:,] = np.argmax(d[0][0])
    translated.append(np.argmax(d[0][0]))

    if dec_inp_seq[0][0] == 1 or len(translated)>16:
      stop = True
  return translated
```
[use this script for more info][link1]




[link1]: <https://github.com/ravis2114/Neural-Machine-Translation/blob/main/encoder_decoder.py>
