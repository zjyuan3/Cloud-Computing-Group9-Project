import os
import pandas as pd
import re
from bs4 import BeautifulSoup
import json
import numpy as np
#将在停用词
stopwords=[]
for stopword in open("stopwords.txt","r").readlines():
    stopwords.append(stopword.strip().replace('\n', ''))

#做数据预处理
def clean_text(text,vocabs):
    clean_words=[]
    text=BeautifulSoup(text,'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]',' ', text)#去掉标点符号
    words = text.lower().split() #转小写
    for w in words:
        if w not in stopwords:
            clean_words.append(w)
            vocabs.add(w)
    words = [w for w in words if w not in stopwords]#去掉应用词
    return ' '.join(words)


#下采样
def lower_sample_data(df, percent=1):
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    '''
    data1 = df[df['label'] == "unsup"]  # 将多数类别的样本放在data1
    data2 = df[df['label'] == "neg"]  # 将多数类别的样本放在data1
    data0 = df[df['label'] == "pos"]  # 将少数类别的样本放在data0
    index = np.random.randint(
        len(data1), size=percent * (len(df) - len(data1)))  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return(pd.concat([lower_data1, data0,data2]))


if __name__ == '__main__':
    vocabs=set()
    path = "train" #文件夹目录
    data={'text':[],"label":[]}
    files= os.listdir(path)
    for file in files:
          for f in os.listdir(os.path.join(path, file)):
                f2 = open(os.path.join(path, file,f),"r",encoding='utf-8')
                line = f2.read()
                line=clean_text(line,vocabs)
                data['text'].append(line)
                data['label'].append(file)
    lower_sample_data(pd.DataFrame(data)).to_excel('train1.xlsx',index=None,header=None)
    # f = open('vocabs.txt','w',encoding='utf-8')
    # f.write(json.dumps(dict(zip(vocabs,range(len(vocabs))))))
