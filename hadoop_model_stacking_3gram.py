import os
import sys
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import re
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


x = []
y = []

def clean_text(text):
    clean_words=[]
    text=BeautifulSoup(text,'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]',' ', text)#去掉标点符号
    words = text.lower().split() #转小写
    return ' '.join(words)

 
def mapper():
    c = 1
    file_name = os.getenv('map_input_file')
    if 'neg' in str(file_name):
        label=0
    elif 'pos' in str(file_name):
        label=1
    for line in sys.stdin:
        print ("%s,%s,%s" %(c,clean_text(line),label))  #增加一个序列号，为了reducer之前排序数据
        c += 1



#========SVM========#
def SvmClass(x_train, y_train):
	clf = SVC(kernel = 'linear',probability=True)#default with 'rbf'
	clf.fit(x_train, y_train)#训练，对于监督模型来说是 fit(X, y)，对于非监督模型是 fit(X)
	return clf

#=====NB=========#
def NbClass(x_train, y_train):
	clf=MultinomialNB(alpha=0.01).fit(x_train, y_train)
	return clf

#========Logistic Regression========#
def LogisticClass(x_train, y_train):
	clf = LogisticRegression(penalty='l2')
	clf.fit(x_train, y_train)
	return clf

#========准确率召回率 ========#
def Precision(svm,lr,nb,x_test,y_test):
	svm_answer = svm.predict_proba(x_test)[:,1]
	lr_answer = lr.predict_proba(x_test)[:,1]
	nb_answer = nb.predict_proba(x_test)[:,1]
	report = (svm_answer+lr_answer+nb_answer )/3 > 0.5
	print(classification_report(y_test, report, target_names = ['neg', 'pos']))
	print('准确率: %.2f' % accuracy_score(y_test, report))


def reducer():
    for line in sys.stdin:
        item = line.rstrip('\n').split(',')
        x.append(item[1])
        y.append(eval(item[2]))
    # print ("%s,%s"%(x,y))
    count_vec = TfidfVectorizer(ngram_range=(3, 3))
    x_train, x_test, y_train, y_test= train_test_split(x, y,stratify=y, test_size = 0.3)
    x_train = count_vec.fit_transform(x_train)
    x_test  = count_vec.transform(x_test)
    print('**************模型融合************  ')
    svm=SvmClass(x_train, y_train)
    lr=NbClass(x_train, y_train)
    nb=LogisticClass(x_train, y_train)
    Precision(svm,lr,nb,x_test,y_test)


d = {'mapper': mapper, 'reducer': reducer}
if sys.argv[1] in d:
    d[sys.argv[1]]()
