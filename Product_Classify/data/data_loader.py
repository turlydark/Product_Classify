import config as cfg
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle
import joblib
import json

def data_loader(test_rate=0.1):
    data = pd.read_csv(r"data/train_cut_.csv")
    data = shuffle(data, random_state = 10)
    test_size = int(test_rate * len(data))

    tokenizer = Tokenizer(num_words=cfg.max_word)
    tokenizer.fit_on_texts(data['ITEM_NAME'][test_size:])

    index_dict = json.dumps(tokenizer.word_index)
    with open("word_distract_dict.json", "w", encoding='utf-8') as f:
        json.dump(index_dict, f)
    joblib.dump(tokenizer, "tok.m")
    #print(tokenizer.word_index)
    train_sequences = tokenizer.texts_to_sequences(data['ITEM_NAME'][test_size:])
    test_sequences = tokenizer.texts_to_sequences(data['ITEM_NAME'][:test_size])
    train_processed = pad_sequences(train_sequences, maxlen=cfg.max_len)
    test_processed = pad_sequences(test_sequences, maxlen=cfg.max_len)

    all_labels = list(set(data['TYPE']))
    label_index = dict(zip(all_labels, range(len(all_labels))))
    labels = [label_index[label] for label in data['TYPE']]
    # 将字典保存为json格式
    label_dict = json.dumps(label_index)
    with open("product_label_dict.json", "w", encoding='utf-8') as f:
        json.dump(label_dict, f)

    labels = to_categorical(labels, num_classes=cfg.num_classes)

    #print(label_index)
    return (train_processed, labels[test_size:]), (test_processed, labels[:test_size])

if __name__ == '__main__':
    (train_processed, train_label), (test_processed, test_label) = data_loader()

    print(train_processed.shape)
    print(train_label.shape)
    print(train_processed[0])
    print(len(train_processed))
    print(len(train_processed[0]))
    print(train_label[0])
    print(train_label[1])
    print(len(train_label))
    print(train_label.shape)

