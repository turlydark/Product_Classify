import pandas as pd
import jieba
import re
import os
import tensorflow as tf
import numpy as np
import keras.backend.tensorflow_backend as KTF
import config as cfg
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import joblib
import json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.preprocessing import sequence
from keras.utils import to_categorical
import config as cfg
from keras.models import Model
from keras.layers import *
from train import get_model
from sklearn.utils import shuffle

def set_gpu():
    print("Set GPU...")
    os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    KTF.set_session(session)

def data_process():
    print("Data Processing...")
    data = pd.read_csv('test.tsv', sep='\t', encoding='gb18030')
    my_data = []

    for num,i in enumerate(data['ITEM_NAME']):
        name = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", i)
        name = jieba.cut(name, cut_all=True)
        my_data.append(" ".join(name))

        if (num + 1) % 10000 == 0:
            print(num)
            # break
    new_data = pd.DataFrame(columns=['ITEM_NAME'])
    new_data['ITEM_NAME'] = pd.Series(my_data, name='ITEM_NAME')
    new_data.to_csv("test_cut_.csv", encoding='utf-8', index=None)

def data_loader():
    print("Data Loading...")
    data = pd.read_csv(r"test_cut_.csv")
    test_size = len(data)
    print("test size is : ", test_size)

    # tokenizer = joblib.load("tok.m")

    word_data = pd.read_csv(r"data/train_cut_.csv")
    tokenizer = Tokenizer(num_words=cfg.max_word)
    tokenizer.fit_on_texts(word_data['ITEM_NAME'])

    test_sequences = tokenizer.texts_to_sequences(data['ITEM_NAME'])
    test_processed = pad_sequences(test_sequences, maxlen=cfg.max_len)

    print(test_processed.shape)
    print(test_processed[0])

    return test_processed

def model(test):
    print("Set Model...")
    model = get_model()
    print("Data Predicting...")
    prediction_class = np.argmax(model.predict(test), axis=1)
    print(prediction_class)
    print(prediction_class.shape)

    return prediction_class


def matrix_to_label(prediction_class):
    print("Matrix to labels...")
    with open("product_label_dict.json", 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    label_dict = json.loads(load_dict)
    print("product label dict is : ", label_dict)

    new_dict = {v: k for k, v in label_dict.items()}
    print(new_dict)
    my_data = []
    for i in prediction_class:
        my_data.append(new_dict[i])

    data = pd.read_csv('test.tsv', sep='\t', encoding='gb18030')
    new_data = pd.DataFrame(columns=['ITEM_NAME','TYPE'])
    new_data['ITEM_NAME'] = data['ITEM_NAME']
    new_data['TYPE'] = pd.Series(my_data, name='TYPE')
    new_data.to_csv("res.tsv", sep='\t', encoding='utf-8', index=None)

if __name__ == '__main__':
    # pass
    #设置gpu
    set_gpu()

    #分词
    data_process()

    #文本转化为数值向量
    test = data_loader()

    #载入模型，预测
    prediction_class = model(test)


    #预测值转换为商品类型
    matrix_to_label(prediction_class)