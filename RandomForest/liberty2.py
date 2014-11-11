#-------------------------------------------------------------------------------
# Name:        liberty.py
# Purpose:      Kaggle competition
#               Predict expected fire losses for insurance policies
#				Example of successful using Random Forest algorithm 	
# Author:      Serge Korsunenko
#
# Created:     26/08/2014
# Copyright:   (c) Serge 2014
# Licence:     use as you want
#-------------------------------------------------------------------------------
from __future__ import division
import sys, csv
import numpy as np
from time import time
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.preprocessing import Imputer,LabelEncoder
from sklearn.linear_model import Ridge,Lasso,SGDRegressor,PassiveAggressiveRegressor
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
from sklearn.svm import SVR,SVC,NuSVR
from sklearn.preprocessing import normalize,OneHotEncoder,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.neighbors import NearestCentroid
from scipy.stats import pearsonr
import pandas as pd
import logging
logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)
"""
This data represents almost a million insurance records and the task is to
 predict a transformed ratio of loss to total insured value
 (called "target" within the data set). The provided features contain
 policy characteristics, information on crime rate, geodemographics, and weather.

The train and test sets are split randomly. For each id in the test set,
 you must predict the target using the provided features.
Download data file from
https://www.kaggle.com/c/liberty-mutual-fire-peril/data
"""

def weighted_gini(act,pred,weight):
    """
    Submissions are evaluated on the normalized, weighted Gini coefficient.
    The weights used in the calculation are represented by var11 n the dataset.
    To calculate the normalized weighted Gini, your predictions are sorted from
    largest to smallest. This is the only step where the explicit prediction
    values are used (i.e. only the order of your predictions matters).
    We then move from largest to smallest, asking "In the leftmost x% of the data,
    how much of the actual observed, weighted loss (the target multiplied by
    the provided weight) have you accumulated?" With no model, you expect to
    accumulate 10% of the loss in 10% of the predictions, so no model
    (or a "null" model) achieves a straight line. The area between your curve
     and this straight line the Gini coefficient.
    """
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})
    df.sort('pred',ascending=False,inplace=True)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    n = df.shape[0]
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

	
def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)


def load_data(filename  = 'train.csv'):
    start_time = time()
    xdata=[]
    ydata=[]
    i=0
    xn="X_"
    yn="Y_"
    if filename[0:3]=='tes':
        xn="Xt_"
        yn="Yt_"
    for e, line in enumerate( open(filename) ):
        if e==0 :
            headers=line.split(",")
            for i,h in enumerate(headers):
                print i,h
            return
            continue
        r=line.split(",")
        if filename[0:3]=='tes':
            ydata.append(r[0])
            xdata.append(r[1:])
        else:
            ydata.append(r[1])
            xdata.append(r[2:])
        if e % 151000==0:
            i += 1
            print e,time() - start_time, "seconds"
            X = np.array(xdata)
            Y =np.array(ydata)
            print xn,X.shape,Y.shape
            xdata=[]
            ydata=[]
            np.save(xn + str(i),X)
            np.save(yn + str(i),Y)
            del X
            del Y
    i += 1
    print e,time() - start_time, "seconds"
    X = np.array(xdata)
    Y =np.array(ydata)
    xdata=[]
    ydata=[]
    np.save(xn + str(i),X)
    np.save(yn + str(i),Y)

	
def prepare_data():
    X1 = np.load("Xt_1.npy")
    X1 = X1[:,0:17]
    X2 = np.load("Xt_2.npy")
    X2 = X2[:,0:17]
    X3 = np.load("Xt_3.npy")
    X3 = X3[:,0:17]
    X = np.vstack((X1,X2,X3))
    del X1
    del X2
    del X3
    for i in range(X.shape[1]):
        ndx = np.where(X[:,i]=='NA')[0]
        X[ndx,i]='0'
        ndx = np.where(X[:,i]=='Z')[0]
        X[ndx,i]='0'
    print X.shape
    le = LabelEncoder()
    for j in range(0,9):
        le.fit(X[:,j])
        print le.classes_
        X[:,j] = le.transform(X[:,j])
    X = np.array(X,dtype=float)
    print X.shape
    np.save("Xvar_t",X)
    print "Save Xvar"
    del X
    Y1=np.load("Yt_1.npy")
    Y2=np.load("Yt_2.npy")
    Y3=np.load("Yt_3.npy")
    Y = np.hstack((Y1,Y2,Y3))
    np.save("Y_t",Y)
    del Y
    print "Save Y"
    X1 = np.load("Xt_1.npy")
    X1 = X1[:,18:64]
    X2 = np.load("Xt_2.npy")
    X2 = X2[:,18:64]
    X3 = np.load("Xt_3.npy")
    X3 = X3[:,18:64]
    X = np.vstack((X1,X2,X3))
    del X1
    del X2
    del X3
    for i in range(X.shape[1]):
        ndx = np.where(X[:,i]=='NA')[0]
        X[ndx,i]='0'
    X = np.array(X,dtype=float)
    print X.shape
    np.save("Xcrgeo_t",X)
    del X
    print "Save Xcrgeo"


def prepare_data2():
    X1 = np.load("Xt_1.npy")
    X1 = X1[:,64:-1]
    print X1.shape
    for i in range(0,X1.shape[1]):
        ndx = np.where(X1[:,i]==u'NA')[0]
        X1[ndx,i]='0.0'

    X1 = X1.astype(np.float)
    print "X1"
    X2 = np.load("Xt_2.npy")
    X2 = X2[:,64:-1]
    for i in range(X2.shape[1]):
        ndx = np.where(X2[:,i]=='NA')[0]
        X2[ndx,i]='0'
    X2 = np.array(X2,dtype=float)
    print "X2"

    X3 = np.load("Xt_3.npy")
    X3 = X3[:,64:-1]
    for i in range(X3.shape[1]):
        ndx = np.where(X3[:,i]=='NA')[0]
        X3[ndx,i]='0'
    X3 = np.array(X3,dtype=float)
    print "X3"
    X = np.vstack((X1,X2,X3))
    del X1
    del X2
    del X3

    np.save("Xweath_t",X)
    del X
    print "Save Xweath"

	
def predict_proba( scores):
    #convert decision_function to probability
    prob = 1. / (1. + np.exp(-scores))
    return prob

	
def binardata(filename="Xvar.npy"):
    #Binarize categories
    #logging.info("Load data...")
    X=np.load(filename)
#    print X.shape
    X[:,9]=0   #dummy
    W=X[:,10].copy()   #weight
    feat = np.array([0,1,2,3,4,5,6,7,8])
    enc = OneHotEncoder(categorical_features=feat)
    Xn = enc.fit_transform(X).toarray()
    return Xn,W

	
def binardata0():
    #Binarize categories
    #logging.info("Load data...")
    X=np.load("Xvar.npy")
    Xt=np.load("Xvar_t.npy")
    t=X.shape[0]
    print X.shape,Xt.shape
    X[:,9]=0   #dummy
    Xt[:,9]=0
    W=X[:,10].copy()   #weight
    feat = np.array([0,1,2,3,4,5,6,7,8])
    enc = OneHotEncoder(categorical_features=feat)
    Xn = enc.fit_transform(X).toarray()
    Xnt = enc.fit_transform(Xt).toarray()
    X = np.vstack((Xn,Xnt))
    scale = MinMaxScaler()
    X = scale.fit_transform(X)
    Xn = X[0:t]
    Xnt = X[t:]
    print Xn.shape,Xnt.shape
    return Xn,W,Xnt

	
def binardata2(filename="Xcrgeo.npy"):
    X = np.load(filename)
    scale = MinMaxScaler()
    Xn = scale.fit_transform(X)
    return Xn


def binardata3(filename="Xweath.npy"):
    X = np.load(filename)
    scale = MinMaxScaler()
    Xn = scale.fit_transform(X)
    return Xn


def train():
    logging.info("Load data...")
    X=np.load("Xvar.npy")
    W=X[:,10].copy()
    Y=np.load("Y.npy")
    Y = Y.astype(np.float)
    ndx=np.where(Y!=0.0)[0]
    print X.shape,ndx.shape
    gf=[]
    tn=int(X.shape[0]*0.68)
    tv=int(X.shape[0]*0.7)
    Xtr = X[0:tn]
    Ytr = Y[0:tn]
    Xts = X[tv:]
    Yts = Y[tv:]
    Wts = W[tv:].copy()
    Xtv = X[tn:tv]
    Ytv = Y[tn:tv]
    Wtv = W[tn:tv]

    Ytr1 = Y[0:tn].copy()
    ndx=np.where(Ytr >0)[0]
    Ytr1[ndx]=1
    logging.info("Train...")
    model = Ridge(alpha=2.0, fit_intercept=False)
#    model = GradientBoostingRegressor(max_depth=3,learning_rate=0.1,loss='ls',n_estimators=30,random_state=2,verbose=1)
    weight={0:0.002,1:1.0}
    clflst=[]
    for i in range(1,2):
#        clf2 = Ridge(alpha=1.0, fit_intercept=True,normalize=False)
#        clf = RandomForestClassifier(max_depth=5,n_estimators=40,n_jobs=2,max_features='auto',random_state=0)
#        clf2 = RandomForestRegressor(max_depth=5,n_estimators=40,n_jobs=2,max_features='auto',random_state=12)
#        clf = LogisticRegression(penalty='l1', C=0.10, tol=0.001,fit_intercept=False,class_weight=weight)
#        clf = RidgeClassifier(alpha=1.0, fit_intercept=True,normalize=False,class_weight=weight)
#        clf = ExtraTreesClassifier(max_depth=8,n_estimators=60, n_jobs=2,random_state=0)
#        clf = SGDClassifier(alpha=0.0000001,loss='hinge',class_weight=weight, n_iter=100,penalty='l1',l1_ratio=0.35)
#        clf = NearestCentroid()
#        clf = KNeighborsClassifier(n_neighbors=100,weights='distance')
#        clf2 = SVR(kernel='linear',verbose=1)
#        clf = Lasso()
#        clf2 = KNeighborsRegressor(algorithm='auto',n_neighbors=100, weights='uniform')
#        clf2 = SGDRegressor()
#        clf=PassiveAggressiveRegressor(class_weight=True)
#        clflst.append(RandomForestRegressor(max_depth=5,n_estimators=40,n_jobs=2,max_features='auto',random_state=12,verbose=1))
        clflst.append( GradientBoostingRegressor(max_depth=4,learning_rate=0.1,loss='ls',n_estimators=60,random_state=0,verbose=1))
#        clflst.append(Ridge(alpha=1.0, fit_intercept=True,normalize=False))
#        clflst.append(Lasso())
#        clflst.append(PassiveAggressiveRegressor(class_weight=True))
#        clflst.append(SGDRegressor(alpha=0.0000001, n_iter=100,penalty='l1'))
        predbs=[]
        predb=[]
        for c in clflst:
            logging.info("Train " + str(c))
            c.fit(Xtr,Ytr)
            predbs.append(c.predict(Xtv))
            predb.append(c.predict(Xts))
##        logging.info("Train " + str(clf))
##        clf.fit(Xtr,Ytr1)
##        if hasattr(clf,"predict_proba"):
##            predbs.append(clf.predict_proba(Xtv)[:,1])
##            predb.append(clf.predict_proba(Xts)[:,1])
##        else:
##            predbs.append(predict_proba(clf.decision_function(Xtv)))
##            predb.append(predict_proba(clf.decision_function(Xts)))

        logging.info("Predict...")
        Xsm=np.vstack(predbs).T
        model.fit(Xsm,Ytv)
        if hasattr(model,"coef_"):
            print model.coef_
        Xsm=np.vstack(predb).T
        pr1 = model.predict(Xsm)
#        pr1 = predb[0]*0.4 + predb[1]*0.6   #+ predb[2]*0.1
#        pr1 = clf.predict(Xts)
#        pr1=(clf.predict(Xts)  +clf1.predict(Xts))/2
##        if hasattr(clf,"predict_proba"):
##            pr1=clf.predict_proba(Xtv)[:,1]
##            pr2=clf.predict_proba(Xts)[:,1]
##        else:
##            pr1 = predict_proba(clf.decision_function(Xtv))
##            pr2 = predict_proba(clf.decision_function(Xts))
#        +clf1.predict(Xts)+clf2.predict(Xts)
        print i,normalized_weighted_gini(Yts,pr1,Wts)
        logging.info("Done.")

def submit():
    logging.info("Submit Load data...")
    X,W,Xt=binardata0()
    Y=np.load("Y.npy")
    Y = Y.astype(np.float)
#    clf = GradientBoostingRegressor(max_depth=4,learning_rate=0.1,loss='ls',n_estimators=60,random_state=0, verbose=1)
    clf2 = RandomForestRegressor(max_depth=5,n_estimators=40,n_jobs=2,max_features='auto',random_state=12, verbose=1)
    print clf2
    clf2.fit(X,Y)
    Y=np.load("Y_t.npy")
    Y = Y.astype(np.float)
    print "predict"
    pr1=clf2.predict(Xt)
    filename_submission = "submission.csv"
    print "Creating submission file", filename_submission
    f = open(filename_submission, "w")
    print >> f, "id,target"
    for i in range(len(Xt)):
        print >> f, str(Y[i]) + "," + str(pr1[i])
    f.close()
    print "Done."



def main():
#    binardata0()
#    train()
#    prepare_data()
#    prepare_data2()
#    load_data()
    submit()

if __name__ == '__main__':
    main()
