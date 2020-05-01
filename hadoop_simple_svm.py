import os
import sys
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
x = []
y = []
def mapper():
    c = 1
    for line in sys.stdin:
        item = line.rstrip('\n').split(",")
        target_names = {'neg':0, 'pos':1}
        print ("%s,%s,%s" %(c,item[0],target_names[item[1]]))  #增加一个序列号，为了reducer之前排序数据
        c += 1


#========Logistic Regression========#
def LogisticClass(x_train, y_train):
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l2')
        clf.fit(x_train, y_train)
        return clf

#========准确率召回率 ========#
def Precision(clf,x_test,y_test):
        doc_class_predicted = clf.predict(x_test)
        print(np.mean(doc_class_predicted == y_test))#预测结果和真实标签
        #准确率与召回率
        precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
        answer = clf.predict_proba(x_test)[:,1]
        report = answer > 0.5
        print(classification_report(y_test, report, target_names = ['neg', 'pos']))

        print('准确率: %.2f' % accuracy_score(y_test, doc_class_predicted))


def reducer():
    for line in sys.stdin:
        item = line.rstrip('\n').split(',')
        x.append(item[1])
        y.append(eval(item[2]))
    # print ("%s,%s"%(x,y))
    count_vec = TfidfVectorizer(binary = False)
    x_train, x_test, y_train, y_test= train_test_split(x, y,stratify=y, test_size = 0.3)
    x_train = count_vec.fit_transform(x_train)
    x_test  = count_vec.transform(x_test)
    print('**************逻辑回归************  ')
    Precision(LogisticClass(x_train, y_train),x_test,y_test)

d = {'mapper': mapper, 'reducer': reducer}
if sys.argv[1] in d:
    d[sys.argv[1]]()
