from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, merge, Dense
from create_dataset import Data_gener

EPOCHS = 10
BATCH_SIZE = 512
VAL_BATCH_SIZE = 4
TRAIN_STEPS = 64
MAX_SEQUENCE_LEN = 20
MAX_NUM_WORDS = 3500

def DL_Model(MAX_NUM_WORDS, MAX_SEQUENCE_LEN):
    embedding_layer = Embedding(MAX_NUM_WORDS, 128, input_length = MAX_SEQUENCE_LEN, trainable = True, name = 'emb')
    conv_layer = Conv1D(256, 5, activation='sigmoid', name = 'conv') 
    sequence_input_1 = Input(shape = (MAX_SEQUENCE_LEN,), dtype='int32' )
    embedded_sequences_1 = embedding_layer(sequence_input_1)
    x_1 = conv_layer(embedded_sequences_1)
    x_1 = GlobalMaxPooling1D()(x_1)  # global max pooling
    x_1 = Dropout(0.5)(x_1)
    sequence_input_2 = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_input_2)
    x_2 = conv_layer(embedded_sequences_2)
    x_2 = GlobalMaxPooling1D()(x_2)  # global max pooling
    x_2 = Dropout(0.5)(x_2)
    x = merge([x_1,x_2],mode = 'concat')
    x = Dense(128,activation = 'sigmoid', name = 'dense_1')(x)
    x = Dropout(0.5)(x)
    pred = Dense(1, activation='sigmoid', name = 'dense_2')(x)

    model = Model(input=[sequence_input_1, sequence_input_2], output=pred)
    return model

g = Data_gener('wine', batch_size = BATCH_SIZE)
gg = g.gener('train','numpy')

model = DL_Model(MAX_NUM_WORDS,MAX_SEQUENCE_LEN)
model.compile(optimizer='rmsprop', loss='mean_squared_error',metrics = [ 'acc'])
print('begin_train')
for epoch in range(EPOCHS):
    model.fit_generator(generator = gg, steps_per_epoch = TRAIN_STEPS, nb_epoch = 1, verbose = 2, use_multiprocessing=True)
gt = g.gener('test','numpy')
tx,tl = next(gt)
ty = model.predict(tx)
import numpy as np
print(np.dot(ty.squeeze(),tl.squeeze()))