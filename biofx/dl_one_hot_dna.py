import numpy as np
def onehote(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq2 = [mapping[i] for i in sequence]
    return np.eye(4)[seq2]

dna='ATTTACGGATTGCTGA'
#calling onehote function
oneHotEncodedDna= onehote(dna)
print(oneHotEncodedDna)



def onehote(seq):
    seq2=list()
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [1., 0., 0., 0.], "T":[0., 0., 0., 1.]}
    for i in seq:
      seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
    return np.array(seq2)

dna="FTGTANATGCGFCGTGCTAA"
oneHotEncodedDna=onehote(dna)
print("dna\n",list(dna))
print('encoded dna\n',oneHotEncodedDna.T)


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def onehote(sequence):
  seq_array = np.array(list(sequence)) 
  #integer encode the sequence
  label_encoder = LabelEncoder()
  integer_encoded_seq = label_encoder.fit_transform(seq_array) 
  #one hot the sequence
  onehot_encoder = OneHotEncoder(sparse=False)
  #reshape because that's what OneHotEncoder likes
  integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
  onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
  return onehot_encoded_seq

dna='ATGATCGCATAGATGACTAG'
print("DNA\n",list(dna))
print("encoded DNA \n",onehote(dna).T)

# DNA --------------------------------------------------
import numpy as np

seq_length = 40
num_sample = 1000

# PFM from JASPAR
motif = np.array([[   0,   2, 104, 104,   1,   2, 103, 102,   0,   0,  99, 105,   0,   0, 100, 102,   5,   3],
                  [   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   4,   0,   0,   2,   3,   0,   0,   3],
                  [ 105, 103,   1,   1, 104, 102,   2,   3, 104, 103,   2,   0, 105, 103,   0,   2,  97,  97],
                  [   0,   0,   0,   0,   0,   1,   0,   0,   1,   0,   0,   0,   0,   0,   2,   1,   3,   2]])


freq = np.hstack([np.ones((4,(seq_length-motif.shape[1])//2)), 
                  motif,
                  np.ones((4,(seq_length-motif.shape[1])//2))])
print('Sequence matrix shape: {}'.format(freq.shape))

#normalize to PWM and generate positive samples
pos = np.array([np.random.choice(['A', 'C', 'G', 'T'], num_sample, p=freq[:,i]/sum(freq[:,i])) 
                for i in range(seq_length)]).transpose()
[''.join(x) for x in pos[1:10,:]]

neg = np.array([np.random.choice(['A', 'C', 'G', 'T'], num_sample, p=np.array([1,1,1,1])/4.0)
                for i in range(seq_length)]).transpose()
[''.join(x) for x in neg[1:10,:]]

pos_tensor = np.zeros(list(pos.shape) + [4])
neg_tensor = np.zeros(list(neg.shape) + [4])

base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

#naive one-hot encoding
for row in range(num_sample):
    for col in range(seq_length):
        pos_tensor[row,col,base_dict[pos[row,col]]] = 1
        neg_tensor[row,col,base_dict[neg[row,col]]] = 1

print('Positive sample matrix shape: {}'.format(pos.shape))
# this should be a 3D tensor with shape: (samples, steps, input_dim)
print('Positive sample tensor shape: {}'.format(pos_tensor.shape))

X = np.vstack((pos_tensor, neg_tensor))
y = np.concatenate((np.ones(num_sample), np.zeros(num_sample)))

print('Training set shape: {}'.format(X.shape))
print('Training set label shape: {}'.format(y.shape))
print('\nOne-hot encoding looks like:\n {}'.format(X[0,0:10,:]))

#here comes the deep learning part
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.activations import relu
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD

model = Sequential()
model.add(Conv1D(1, 19, padding='same', input_shape=(seq_length, 4), activation='relu'))

#sanity check for dimensions
print('Shape of the output of first layer: {}'.format(model.predict_on_batch(pos_tensor[0:32,:,:]).shape))


#model.add(MaxPooling1D(pool_length=4))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X, y, validation_split=0.2, epochs=10)  # starts training


#have a look at the filter
convlayer = model.layers[0]
weights = convlayer.get_weights()[0].squeeze()
print('Convolution parameter shape: {}'.format(weights.shape))


num2seq = ['A','C','G','T']

''.join([num2seq[np.argmax(weights[i,:])] for i in range(weights.shape[0])])