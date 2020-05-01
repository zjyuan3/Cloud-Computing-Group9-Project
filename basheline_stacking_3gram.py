import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import  numpy as np

#========SVM========#
def SvmClass(x_train, y_train):
	from sklearn.svm import SVC
	#调分类器
	clf = SVC(kernel = 'linear',probability=True)#default with 'rbf'
	clf.fit(x_train, y_train)#训练，对于监督模型来说是 fit(X, y)，对于非监督模型是 fit(X)
	return clf

#=====NB=========#
def NbClass(x_train, y_train):
	from sklearn.naive_bayes import MultinomialNB
	clf=MultinomialNB(alpha=0.01).fit(x_train, y_train)
	return clf

#========Logistic Regression========#
def LogisticClass(x_train, y_train):
	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression(penalty='l2')
	clf.fit(x_train, y_train)
	return clf


#========准确率召回率 ========#
def Precision(clf):
	doc_class_predicted = clf.predict(x_test)
	print(np.mean(doc_class_predicted == y_test))#预测结果和真实标签
	#准确率与召回率
	precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
	answer = clf.predict_proba(x_test)[:,1]
	report = answer > 0.5
	print(classification_report(y_test, report, target_names = ['neg', 'pos']))
	print("--------------------")
	from sklearn.metrics import accuracy_score
	print('准确率: %.2f' % accuracy_score(y_test, doc_class_predicted))
#========准确率召回率 ========#
def Precision1(svm,lr,nb):
	svm_answer = svm.predict_proba(x_test)[:,1]
	lr_answer = lr.predict_proba(x_test)[:,1]
	nb_answer = nb.predict_proba(x_test)[:,1]
	report = (svm_answer+lr_answer+nb_answer )/3 > 0.5
	print(classification_report(y_test, report, target_names = ['neg', 'pos']))
	from sklearn.metrics import accuracy_score
	print('准确率: %.2f' % accuracy_score(y_test, report))


if __name__ == '__main__':
    target_names = {'neg':0, 'pos':1}
    df=pd.read_csv("train.csv",delimiter=",",names=['text', 'label'])
    data=df['text']
    labels=df['label']
    x=np.array(data)
    labels=np.array(labels)
    labels=[int (target_names[i])for i in labels]
    movie_target=labels
    #转换成空间向量
    count_vec = TfidfVectorizer(ngram_range=(3, 3))
    #加载数据集，切分数据集80%训练，20%测试
    x_train, x_test, y_train, y_test= train_test_split(x, movie_target,stratify=movie_target, test_size = 0.3)
    x_train = count_vec.fit_transform(x_train)
    x_test  = count_vec.transform(x_test)
    svm=SvmClass(x_train, y_train)
    lr=NbClass(x_train, y_train)
    nb=LogisticClass(x_train, y_train)
    Precision1(svm,lr,nb)

