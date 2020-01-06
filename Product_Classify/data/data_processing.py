import pandas as pd
import jieba
import re

def data_diastract():
    data = pd.read_csv(r'train.tsv', sep='\t', encoding='gb18030')
    my_data = []
    for num, (i, j) in enumerate(zip(data['ITEM_NAME'], data['TYPE'])):
        name = jieba.cut(i, cut_all=True, HMM = True)
        my_data.append(" ".join(name))

        if (num+1) % 1000 == 0:
            print(num)
            #break

    new_data = pd.DataFrame(columns=['ITEM_NAME', 'TYPE'])

    new_data['ITEM_NAME'] = pd.Series(my_data, name='ITEM_NAME')
    new_data['TYPE'] = data['TYPE']
    new_data.to_csv("train_cut_.csv", encoding='utf-8', index=None)
if __name__ == '__main__':
    data_diastract()