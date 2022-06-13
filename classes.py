#Importing all the frameworks we will need to import process and predict.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.neural_network import MLPClassifier


class ReadfilExport:
    def readFile(self, url):
        return pd.read_csv(url)

    def exportfile(self,df,fileName):
        # df1 = pd.read_hdf("heart_problem.h5", "/data");
        df.to_csv(fileName, sep='\t')
        #df.to_hdf(fileName, "/data")




class Preprocess:
    
    def __init__(self, dataset):
        self.dataset = dataset
#        self.columns: list = []
        self.nominal_features = ['cp', 'slp', 'thall']
        self.testdict = dict(zip(self.nominal_features, ['CP', 'SL', 'TH']))
            
    def onehot_encode(self):
        df = self.dataset.copy() # makin a df copy so as not to manipulate the existing data set
        for column, prefix in self.testdict.items():
            dummies = pd.get_dummies(df[column], prefix=prefix)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
        return df
    
    def preprocess_inputs(self, df, scaler):
        
        df = df.copy()

        # Split df into X and y
        y = df['output'].copy()
        X = df.drop('output', axis=1).copy() # where we are dropping the target column to predict
        
        # Scale X
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X, y






class TrainDataClass:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def splitData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=0)
        return  X_train, X_test, y_train, y_test

    def modelSVC(self,X_train, y_train, X_test, y_test):
        svm_model = svm.SVC()
        svm_model.fit(X_train, y_train)
        accuracyScore = svm_model.score(X_test, y_test) * 100
        return accuracyScore
        
    def model_Lr(self, X_train, y_train, X_test, y_test):
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        accuracyScore = lr_model.score(X_test, y_test) * 100
        return accuracyScore

    def modelKNN(self,X_train, y_train, X_test, y_test):
            Knn_model = KNeighborsClassifier()
            Knn_model.fit(X_train, y_train)     
            accuracyScore = Knn_model.score(X_test, y_test) * 100
            return accuracyScore

