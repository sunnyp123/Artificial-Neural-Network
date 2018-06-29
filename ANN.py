# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:10:34 2018

@author: Sunny Parihar
"""
#Artificial Neural Network. 
#Data Preprocessing.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing the dataset.
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,[3,4,5,6,7,8,9,10,11,12]].values
X = pd.DataFrame(X)
Y = dataset.iloc[:,13].values
#Encoding the categorical data.
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lebelencoder= LabelEncoder()
X.values[:,1] = lebelencoder.fit_transform(X.values[:,1])
labelencoder = LabelEncoder()
X.values[:,2] = labelencoder.fit_transform(X.values[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Remove Dummy Varibale Trap.
X = X[:,1:]
#Splitting the dataset into the training set and test set.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state =0)
#Feature Scaling (Because of more calculations than others)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#Artificial neural network.
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initializing the Artificial Neural Network.
classifier = Sequential()
#Adding the input layer and first hidden layer.
classifier.add(Dense(output_dim = 6,init = 'uniform',activation='relu',input_dim=11))
#Adding the second hidden layer.
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#Adding the output layer.
classifier.add(Dense(output_dim=1,init = 'uniform',activation='sigmoid'))
#Compiling the ANN.

