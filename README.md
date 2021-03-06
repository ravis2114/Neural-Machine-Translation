# Neural-Machine-Translation
Project translates english language to hindi language in devanagari script. same script can be used for translating any one language to subsequent another language.

### english to hindi example
----
```
actual eng text :  ['the holy dip is on november']
actual hindi text :  ['मुख्य स्नान नवंबर को है']
translated hindi text :  ['नवंबर को अंतिम रूप से ये संशोधित श्रृंखला शुरू होती है।']
```
well it's not perfect since it is trained on just 69201 pairs of english-hindi parallel dataset. To train good encoder-decoder translation model it requires millions of parallel corpus of dataset and good amout of training time.

### Main model architecture to be trained
----
![Main Model to be trained](https://github.com/ravis2114/Neural-Machine-Translation/blob/main/model_images/model.png)

### Encoder Model generated from saved Model
----
![Encoder Model generated from saved Model](https://github.com/ravis2114/Neural-Machine-Translation/blob/main/model_images/encoder_model.png)

### Decoder Model generated from saved Model
----
![Decoder Model generated from saved Model](https://github.com/ravis2114/Neural-Machine-Translation/blob/main/model_images/decoder_model.png)


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
### or
[open this colab notebook][link2]




[link1]: <https://github.com/ravis2114/Neural-Machine-Translation/blob/main/encoder_decoder.py>
[link2]: <https://colab.research.google.com/github/ravis2114/Neural-Machine-Translation/blob/main/neural_machine_translation.ipynb>

License
----
MIT
