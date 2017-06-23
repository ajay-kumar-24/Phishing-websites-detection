from sklearn.datasets import load_boston
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import plotly as py
import pylab
import itertools
import timeit,time
from svmutil import *
import scipy.io
import copy
from scipy.stats import norm
import time

def calculateTheta(X1,Y,flag):
	X = X1
	if flag!=1:
		X = np.insert(X,0,1,axis=1)
	else:
		X = np.insert(X,0,1,axis=1)[:,[0]]
	return np.dot(np.dot(pinv(np.dot(np.transpose(X),X)),(np.transpose(X))),Y)

def plotHistogram(data,index):
	plt.figure("Feature : G "+str(index))
	plt.hist(data,bins = 10)
	plt.title("Feature  G"+str(index))
	plt.show()

def meanSquareError(theta,X,Y,flag):
	sqr = 0
	if flag!=2:
		X = np.concatenate(((np.ones((len(X),1))),X),axis =1)
	for i in range(0,len(Y)):
		if(flag == 0):
			sqr += pow(Y[i]-np.dot(np.transpose(theta),X[i]),2)
		elif flag==2:
			sqr += pow((Y[i]-1),2)
		else:
			X=X[:,[0]]
			sqr += pow(Y[i]-np.dot(np.transpose(theta),X[i]),2)
	return sqr
	

def calculateX(x,n):
	new_x = copy.copy(x)
	for i in range(3,n):
		new_x=np.concatenate((new_x,np.power(x,i-1)),axis =1)
	return new_x

def X(x,i):
	data = np.ones((i+1,1))
	for j in range(1,len(data)):
		data[j]= data[j-1]*x
	return data 

def calculateVariance(theta,h):
	tempList =[]
	for i in range(0,500):
		distribution = (np.random.uniform(-1,1.000000001,size=1))
		tempList.append(distribution)
	for g in range(3,4):
		gsum=0.0
		for r in theta:
			bias =0.0
			for i in range(0,500):
				x = tempList[i]
				y=[]
				tx = X(copy.copy(tempList[i]),h-2)
				hx = np.dot(np.transpose(r),tx)
				for t in theta:
					y.append(np.dot(np.transpose(t),tx))
				ehx = np.array(y).mean()
				a = np.square(hx-ehx)
				b = norm.pdf(calc_fx(x),2*x*x,scale=np.sqrt(0.1))
				c =(1/500.0)
				bias+= a*b*c
			gsum+=bias
		return gsum[0]/100.0 

def calculateBias(theta,h):
	tempList=[]
	for i in range(0,500):
		distribution = (np.random.uniform(-1,1.000000001,size=1))
		ones = np.ones((1,1))
		ones[0] = ones[0]*distribution
		tempList.append(ones)
	for g in range(3,4):
		bias =0.0
		for i in range(0,500):
			x = tempList[i]
			y=[]
			if h!=8:
				cd = X(copy.copy(tempList[i]),h-2)
				for t in theta:
					y.append(np.dot(np.transpose(t),cd))
				ehx = np.array(y).mean()
			if h==8:
				ehx=1.0
			bias += np.square(ehx-(2*x*x))*norm.pdf(calc_fx(tempList[i]),2*x*x,scale=np.sqrt(0.1))*(1/500.0)
		return bias[0]

def bias_A(nos):
	tempList = []
	bias_list=[]
	var_list=[]
	mse_list = []
	for i in range(0,100):
		ones = (np.ones((nos,1)))
		distribution = (np.random.uniform(-1,1.000000001,size=nos))
		for i in range(0,len(ab)):
			ones[i] = ones[i]*distribution[i]
		tempList.append(ones)
	data = (tempList)
	y = []
	theta_list = []
	for i in data:
		arr  = calc_fx(copy.deepcopy(i))
		ones = (np.ones((nos,1)))
		for j in range(0,len(arr)):
			ones[j] = ones[j]*arr[j]
		y.append(ones)
	Y=y
	mse = []
	for i in range(0,len(data)):
		mse.append(meanSquareError(1,data[i],(Y[i]),2))
	bias_list.append(calculateBias([],8))
	var_list.append(0)

	mse_list.append(np.array(mse).sum())
	plotHistogram(np.array(mse),1)
	mse= []
	
	for i in range(0,len(data)):
		theta  = calculateTheta(copy.copy(data[i]),((Y[i])),1)
		theta_list.append(theta)
		mse.append(meanSquareError(copy.copy(theta),data[i],(Y[i]),1))
	bias_list.append((calculateBias((copy.copy(theta_list)),2))) 
	var_list.append(calculateVariance((copy.copy(theta_list)),2))
	plotHistogram(np.array(mse),2)
	mse_list.append(np.array(mse).sum())

	for r in range(3,7):
		mse = []
		theta_list = []
		for i in range(0,len(data)):
			final_X = calculateX(copy.copy(data[i]),r)
			theta  = calculateTheta(copy.copy(final_X),((Y[i])),2)
			theta_list.append(theta)
			mse.append(meanSquareError((theta),final_X,(Y[i]),0))
		bias_list.append(calculateBias(copy.copy(theta_list),r))
		var_list.append(calculateVariance(copy.copy(theta_list),r))
		plotHistogram(np.array(mse),r)
		mse_list.append(np.array(mse).sum())
	dislay('meanSquareError ',mse_list,0)
	dislay('Bias',bias_list,0)
	dislay('Variance',var_list,0)
	

def dislay(string,tempList,counter):
	print
	print string
	for i in temp:
		print 'G'+str(counter)+'  \t = '+str(i)
		counter = counter+1
	print 
	

def fx(x):
	return (2*x*x + np.random.normal(0,np.sqrt(0.1)))

def calc_fx(x):
	y=[]
	for i in range(0,len(x)):
		y.append(fx(x[i]))
	return np.array(y)


features = [1, 6, 7, 13, 14, 25, 28]

def app(list1,list2,list3,a,b,c):
	list1.append(a)
	list2.append(b)
	list3.append(c)


def transformation(x):
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

def transformTarget(x):
	for i in range(0,len(x[0])):
		if x[0][i]==-1:
			x[0][i]=0
	return x

def ridge_theta(trainD,trainT,lm):
	X = np.insert(trainD,0,1,axis=1)
	identity = np.identity(X.shape[1])
	identity[0][0] = 0
	return np.dot(pinv(np.dot(np.transpose(X),X)+lm*identity),np.dot((np.transpose(X)),trainT))
		
def ridge_out(lam,mse,bias,var):
	for i in range(0,len(lam)):
		print
		print 'For Lambda : ',lam[i]
		print 'Sum Squared Error : ',mse[i]
		print 'Bias : ',bias[i]
		print 'variance : ',var[i]

def ridge(nos):
	print 'RIDGE REGRESSION'
	bias_list=[]
	var_list=[]
	l=[]
	mse_list = []
	for i in range(0,100):
		ones = (np.ones((nos,1)))
		ab = (np.random.uniform(-1,1.000000001,size=nos))
		for i in range(0,len(ab)):
			ones[i] = ones[i]*ab[i]
		l.append(ones)
	data = (l)
	y = []
	for i in data:
		arr  = calc_fx(copy.deepcopy(i))
		ones = (np.ones((nos,1)))
		for j in range(0,len(arr)):
			ones[j] = ones[j]*arr[j]
		y.append(ones)
	Y=y
	for r in [0.001,0.003,0.01,0.03,0.1,0.3,1.0]:
		mse = []
		theta_list = []
		for i in range(0,len(data)):
			final_X = calculateX(copy.copy(data[i]),4)
			theta  = ridge_theta(copy.copy(final_X),((Y[i])),r)
			theta_list.append(theta)
			mse.append(meanSquareError((theta),final_X,(Y[i]),0))
		bias_list.append(calculateBias(copy.copy(theta_list),4))
		var_list.append(calculateVariance(copy.copy(theta_list),4))
		mse_list.append(np.array(mse).sum())
	ridge_out([0.001,0.003,0.01,0.03,0.1,0.3,1.0],mse_list,bias_list,var_list)
	

def linear_svm(train,tr_target,test,te_target):
	ACC = 0
	lst = []
	print 'LINEAR SVM'
	for c in [-6,-5,-4,-3,-2,-1,0,1,2]:
		t1 = time.clock()
		accuracy = svm_train(tr_target[0].T.tolist(),train.tolist(),"-c "+str(pow(4,c))+" -q -t 0 -v 3")
		print 'C: 4^'+str(c)+' Kernel: linear Train Time:'+str(time.clock()-t1)
		print '->'
		if accuracy > ACC:
			ACC = accuracy
			C = c

	print "Best c "+str(C)+" and accuracy is "+str(ACC)

def poly_SVM(train,tr_target):
    ACC = 0
    print 'POLYNOMIAL KERNEL'
    for c in [-3,-2,-1,0,1,2,3,4,5,6,7]:
    	for d in [1,2,3]:
    		t1 = time.clock()
    		accuracy = svm_train(tr_target[0].T.tolist(),train.tolist(),"-c "+str(pow(4,c))+" -q -v 3 -t 1 -d "+str(d))
    		print 'C: 4^'+str(c)+' Degree:'+str(d)+'    Kernel:polynomial    Train Time:'+str(time.clock()-t1)
    		print '->'
        	if accuracy > ACC:
        		ACC = accuracy
        		C = c
        		D = d
    print "best c "+str(C)+" best d "+str(D)+" and accuracy is"+str(ACC)


def RBFSVM(train,tr_target):
    ACC = 0
    for c in [-3,-2,-1,0,1,2,3,4,5,6,7]:
    	for g in [-7,-6,-5,-4,-3,-2,-1]:
    		t1 = time.clock()
    		accuracy = svm_train(tr_target[0].tolist(),train.tolist(),"-c "+str(pow(4,c))+" -v 3 -q -t 2 -g "+str(pow(4,g)))
    		print 'C: 4^'+str(c)+'Gamma: 4^'+str(g)+'    Kernel:rbf    Train Time:'+str(time.clock()-t1)
    		print '->'
    		if accuracy > ACC:
    			ACC = accuracy
    			C = c
    			G = g
    print "best c "+str(C)+" best g "+str(G)+" and acc is "+str(ACC)

def SVM():
	train = scipy.io.loadmat("phishing-train")
	test = scipy.io.loadmat("phishing-test")
	train_= train["features"]
	train_t = train["label"]
	test_ = test["features"]
	test_t = test["label"]
	transformTrain =transformation(copy.copy(train_))
	transformTest = transformation(copy.copy(test_))
	transformTrain_tar=transformTarget(copy.copy(train_t))
	transformTest_tar=transformTarget(copy.copy(test_t))
	linear_svm(transformTrain,transformTrain_tar,transformTest,transformTest_tar)
	poly_SVM(transformTrain,transformTrain_tar)
	print 'RBF KERNEL'
	RBFSVM(transformTrain,transformTrain_tar)

if __name__=="__main__":
	bias_A(10)
	#bias_A(100)
	#ridge(100)
	SVM()
	
