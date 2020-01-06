from keras import layers
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.preprocessing import sequence
from keras.utils import to_categorical
import config as cfg
from keras.models import Model
from keras.layers import *
from attention import Position_Embedding, Attention


def built_attention_model():
    S_inputs = Input(shape=(None,), dtype='int32')
    embeddings = Embedding(cfg.max_word, 128)(S_inputs)
    embeddings = Position_Embedding()(embeddings)  # 增加Position_Embedding能轻微提高准确率
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.2)(O_seq)
    outputs = Dense(cfg.num_classes, activation='sigmoid')(O_seq)
    model = Model(inputs=S_inputs, outputs=outputs)
    print(model.summary())
    return model


def build_model():
    model = Sequential()
    model.add(layers.Embedding(cfg.max_word, cfg.word_dim, input_length=cfg.max_len))
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(cfg.num_classes, activation='softmax'))
    return model



def bulit_cnn_modle():
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                         kernel_size,
                                          padding='valid',
                                                           activation='relu',
                                                                            strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1252))
    model.add(Activation('sigmoid'))
