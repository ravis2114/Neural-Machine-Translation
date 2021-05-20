import pandas as pd 
import numpy as np
import re
import string

#load dataset
data = pd.read_csv('dataset.csv', encoding='utf-8')


#clean dataset
def clean_data(data):
	#drop null values
	data = data.dropna()
	#drop duplicates
	data = data.drop_duplicates(subset='english_sentence')
	data = data.drop_duplicates(subset='hindi_sentence')
	#lowercase english sentences
	data.english_sentence = data.english_sentence.apply(lambda x : x.lower())
	#remove ' and " from english and hindi sentences
	data.english_sentence = data.english_sentence.apply(lambda x: re.sub("'", '', x))
	data.english_sentence = data.english_sentence.apply(lambda x: re.sub('"', '', x))
	data.hindi_sentence = data.hindi_sentence.apply(lambda x: re.sub("'", '', x))
	data.hindi_sentence = data.hindi_sentence.apply(lambda x: re.sub('"', '', x))
	#remove punctuation from english and hindi sentences
	special_characters = set(string.punctuation)
	data.english_sentence = data.english_sentence.apply(lambda x: ''.join(char for char in x if char not in special_characters))
	data.hindi_sentence = data.hindi_sentence.apply(lambda x: ''.join(char for char in x if char not in special_characters))
	#remove digits from english and hindi sentences
	num_digits= str.maketrans('','', string.digits)
	data.english_sentence = data.english_sentence.apply(lambda x: x.translate(num_digits))
	data.hindi_sentence = data.hindi_sentence.apply(lambda x: x.translate(str.maketrans('', '', string.ascii_letters + string.digits + string.punctuation + '०१२३४५६७८९')))
	#strip sentences
	data.english_sentence = data.english_sentence.apply(lambda x: x.strip())
	data.hindi_sentence = data.hindi_sentence.apply(lambda x: x.strip())
	#select length of both the sentences, here 0 to 15 is selected
	data = data[(data.hindi_sentence.apply(lambda x : 0<len(x.split())<=15)) & (data.english_sentence.apply(lambda x : 0<len(x.split())<=15))]
	#adding START and END tag and creating an util coloumn
	data['hindi_sentence_util'] = data.hindi_sentence
	data.hindi_sentence = data.hindi_sentence.apply(lambda x : 'START ' + x + ' END')
	#once again checking for duplicates
	data = data.drop_duplicates(subset='english_sentence')
	data = data.drop_duplicates(subset='hindi_sentence')
	data = data.drop_duplicates(subset='hindi_sentence_util')

data = clean_data(data)

#creating word vocab for both the languages
eng_word_vocab = set()
eng_sent_len = []
for sent in data.english_sentence:
    eng_sent_len.append(len(sent.split()))
    for word in sent.split():
        if word not in eng_word_vocab:
            eng_word_vocab.add(word)
hin_word_vocab = set()
hin_sent_len = []
for sent in data.hindi_sentence:
    hin_sent_len.append(len(sent.split()))
    for word in sent.split():
        if word not in hin_word_vocab:
            hin_word_vocab.add(word)
eng_word_vocab = sorted(eng_word_vocab)
hin_word_vocab = sorted(hin_word_vocab)
print('max sent len in eng : ', max(eng_sent_len))
print('max sent len in hin : ', max(hin_sent_len))


#Tokenizing both the laguages vocab
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer_eng = Tokenizer()
tokenizer_hin = Tokenizer()
tokenizer_eng.fit_on_texts(eng_word_vocab)
tokenizer_hin.fit_on_texts(hin_word_vocab)

#converting sentences to vector using tokenization
sequence_eng = tokenizer_eng.texts_to_sequences(data.english_sentence)
sequence_hin = tokenizer_hin.texts_to_sequences(data.hindi_sentence_util)

#padding the tokenized vectors
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_eng = pad_sequences(sequence_eng, padding='post', maxlen=max(eng_sent_len))
padded_hin = pad_sequences(sequence_hin, padding='post', maxlen=max(hin_sent_len)-2)

#creating hindi input and output vectors for teacher's forcing method in encoder-decoder
import numpy as np
val_start = np.zeros((padded_hin.shape[0], 1), dtype=padded_hin.dtype)
val_start[:,:] = 2
val_end = np.zeros((padded_hin.shape[0], 1), dtype=padded_hin.dtype)
val_end[:,:] = 1

padded_hin_in = np.hstack((val_start, padded_hin)).astype(np.int32)
padded_hin_out = np.hstack((padded_hin, val_end)).astype(np.int32)


#Model creation--------------
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

def create_model():
  #encoder
  input_eng = Input(shape=(None,), name='eng_inp_tensor')
  encoder_emb = Embedding(input_dim=len(tokenizer_eng.word_index)+1, output_dim=128, embeddings_initializer='random_normal')(input_eng)
  enc_outputs, enc_state_h, enc_state_c = LSTM(64, return_state=True, kernel_initializer='random_normal')(encoder_emb) #output, hidden states of only last LSTM nodes are returned

  #decoder
  input_hin = Input(shape=(None,), name='hin_inp_tensor')
  decoder_emb = Embedding(input_dim = len(tokenizer_hin.word_index)+1, output_dim=128, embeddings_initializer='random_normal')(input_hin)
  dec_output, dec_state_h, dec_state_c = LSTM(64,return_sequences=True, return_state=True, kernel_initializer='random_normal')(decoder_emb, initial_state=[enc_state_h, enc_state_c]) #initial_state links decoder to encoder and output of all the LSTM nodes are returned, , hidden states of only last LSTM node is returned
  dec_outputs = Dense(len(tokenizer_hin.word_index), activation='softmax', kernel_initializer='random_normal')(dec_output)
  #Model
  model = Model([input_eng, input_hin], dec_outputs)
  return model

#Training Model in TPU setup
# creating model inside TPU
with strategy.scope():
  model = create_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1)
  model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

  #load model if already trained
  # model = load_model('/content/drive/MyDrive/Colab Notebooks/NMT/nmt.h5') #no need to use compile as it is already compiled, if we want to change opt and loss then we can use compile and it will not affect pretrained weight.
  # loss = tf.keras.losses.SparseCategoricalCrossentropy()
  # opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.59, beta_2=0.8999, clipnorm=0.85)
  # model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

#Plotting the model architecture
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,)

#Train the Model
history = model.fit(x=[padded_eng, padded_hin_in],
          y=padded_hin_out,
          batch_size=64,
          epochs=5,
          validation_split=0.3)


#plotting the accuracies and loss metrices
import matplotlib.pyplot as plt
fig, axis = plt.subplots(2,2)
fig.set_size_inches(10,10)
axis[0,0].plot(history.history['accuracy'])
axis[0,0].set_title('train accuracy')
axis[0,1].plot(history.history['loss'])
axis[0,1].set_title('train loss')
axis[1,0].plot(history.history['val_accuracy'])
axis[1,0].set_title('val accuracy')
axis[1,1].plot(history.history['val_loss'])
axis[1,1].set_title('val loss')
plt.show()

#save model
model.save('nmt.h5')

#below codes can also be written in a separate script, you just need to repeat tokenization process to tokenize and pad the sentences before passing them to encoder.
#now comes the translating part
#load the model
from tensorflow.keras.models import load_model
model_load = load_model('nmt.h5')

#visualize the layers
print(model_load.layers)
'''
[<tensorflow.python.keras.engine.input_layer.InputLayer at 0x7f420958b5d0>,
 <tensorflow.python.keras.engine.input_layer.InputLayer at 0x7f420748f790>,
 <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f4208d1b950>,
 <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f420957d390>,
 <tensorflow.python.keras.layers.recurrent_v2.LSTM at 0x7f42086e22d0>,
 <tensorflow.python.keras.layers.recurrent_v2.LSTM at 0x7f42074a3450>,
 <tensorflow.python.keras.layers.core.Dense at 0x7f4208a29310>]
'''
#create encoder model from model_load
inf_enc_model = Model(inputs=model_load.layers[0].output, outputs=model_load.layers[4].output)

#create decoder model from model_load
dec_input_state_h = Input(shape=(None,))
dec_input_state_c = Input(shape=(None,))
dec_inputs_states = [dec_input_state_h, dec_input_state_c]
dec_out, dec_h, dec_c = model_load.layers[5](model_load.layers[3].output, initial_state=dec_inputs_states)

dec_output = model_load.layers[6](dec_out)

dec_model_final = Model(inputs=[model_load.layers[1].output, dec_inputs_states], outputs=[dec_output, [dec_h, dec_c]])

#using encoder and decoder model to translate using recursive model one by one
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

#one demo translation
demo = translate(padded_eng[5])
print('actual eng text : ', tokenizer_eng.sequences_to_texts([padded_eng[5]]))
print('actual hindi text : ', tokenizer_hin.sequences_to_texts([padded_hin[5]]))
print()
print('translated hindi text : ', tokenizer_hin.sequences_to_texts([demo[:-1]])) #last one is end so tranucated

'''
actual eng text :  ['the holy dip is on november']
actual hindi text :  ['मुख्य स्नान नवंबर को है']

translated hindi text :  ['नवंबर को अंतिम रूप से ये संशोधित श्रृंखला शुरू होती है।']
'''

#translating custom sentences
def custom_trans(sent):
  seq = tokenizer_eng.texts_to_sequences(sent)
  padded_seq = pad_sequences(seq, padding='post', maxlen=max(eng_sent_len))
  hin_out = translate(padded_seq[0])
  print('the eng sent is : ', sent)
  print('the translated hin sent is :', tokenizer_hin.sequences_to_texts([hin_out[:-1]]))


custom_trans(['dwarka is also a dhaam among the chaar dhaams'])
'''
the eng sent is :  ['dwarka is also a dhaam among the chaar dhaams']
the translated hin sent is : ['द्वारका में ये गांव को मूल्य उपलब्ध होगा।']
'''