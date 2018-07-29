from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop # optimize argorythm
from keras.utils.data_utils import get_file # to use file 
import numpy as np
import random
import sys # to use file path

path = './gingatetsudono_yoru.txt'
bindata = open(path,"rb").read() #"rb" means read binary 
text = bindata.decode("shift_jis") # decode as string type data, cos txt data is encoded by shit_jis, decode by shift_jis
print('Size of text: ', len(text))
chars = sorted(list(set(text)))

# print(chars)
# print('Total chars:', len(chars))

# make dictionary char to index
char_indices = dict((c,i) for i,c in enumerate(chars)) # enumerates set val with index

# print(char_indices)
# make dictionary index to char
indices_char = dict((i,c) for i,c in enumerate(chars)) 

# print(indices_char)

maxlen = 40

step = 3

sentences = []

next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i+maxlen])

# print(sentences)

# print(next_chars)

# print('Numbe of sentences: ', len(sentences))

# Let Vectornize text like one-hot-vector
# if it contains the char, set flag as 1
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) # set input data
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) # set output data

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]]=1
    y[i, char_indices[next_chars[i]]]=1

# define Model
model = Sequential()
model.add(LSTM(128,input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars))) # Dense : make Neural Network using all layer
model.add(Activation('softmax')) #softmax : convert output value 0 to 1
optimizer = RMSprop(lr=0.01) # RMSprop argorythm is suitable for RNN
model.compile(loss='categorical_crossentropy', optimizer=optimizer) # compile declear start learning  loss: loss function categorical cross entropy indicates how far from true value

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for iteration in range(1,60):
    print()
    print('-'*50)
    print('繰り返し回数: ', iteration)
    model.fit(X, y, batch_size=128, epochs=1) # set data and start learn

    start_index = random.randint(0,len(text)-maxlen-1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('-----diversity' ,diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Seedを生成しました: "' + sentence + '"')
        sys.stdout.write(generated)

        # predict next sentence and add sentence
        for i in range(400):
            x = np.zeros((1,maxlen,len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            
            preds = model.predict(x, verbose=9)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

        print()
        













