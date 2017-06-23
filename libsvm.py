import time
from svmutil import *
import scipy.io
import numpy as np
from scipy.stats import norm
features = [1, 6, 7, 13, 14, 25, 28]

def transform(x):
	for i in features:
		zero = np.zeros((len(x),1))
		one = np.zeros((len(x),1))
		minus = np.zeros((len(x),1))
		for j in range(len(x)):
			if x[:,i][j]==1:
				one[j] = 1
			elif x[:,i][j]==0:
				zero[j]=1
			else:
				minus[j]=1
		x=np.concatenate((x,zero),axis = 1)
		x=np.concatenate((x,one),axis = 1)
		x=np.concatenate((x,minus),axis = 1)
	return np.delete(x,features,axis=1)

def trans_tar(x):
	for i in range(0,len(x[0])):
		if x[0][i]==-1:
			x[0][i]=0
	return x

def RBFSVM(train,tr_target,test,test_target):
	model = svm_train(tr_target[0].tolist(),train.tolist(),"-q -c "+str(pow(4,5))+" -t 2 -g "+str(pow(4,-2)))
	temp_res,test_acc,value  = svm_predict(test_target[0].tolist(),test.tolist(),model)
	print 'Testing Accuracy',test_acc[0]
	 
def SVM():
	train = scipy.io.loadmat("phishing-train")
	test = scipy.io.loadmat("phishing-test")
	train_= train["features"]
	train_t = train["label"]
	test_ = test["features"]
	test_t = test["label"]
	vish_train =transform(train_)
	vish_test = transform(test_)
	vish_train_tar=trans_tar(train_t)
	vish_test_tar=trans_tar(test_t)
	RBFSVM(vish_train,vish_train_tar,vish_test,vish_test_tar)

if __name__ =="__main__":
	SVM()
