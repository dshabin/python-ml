# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3 : 13].values
y = dataset.iloc[:, 13].values

#As we have some categorical variables(encodeing categorical dataset)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#encoding first variable
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

#encoding second variable
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])


onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# ---> preproccess end

# part 2 - make the ANN!
# importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
# selectthe nodes of the hidden layer by average of input and
# output layers -> 11 + 1 = 12 /2 = 6
# select rectifier activation function for inputlayer
# and select sigmoid function for input layer
# rectifoer function = relu
# input dim = our number of independet variable
#first hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform', activation = 'relu',input_dim = 11 ))

#second hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform', activation = 'relu'))

#output layer
# we want to have probability
# change activation function to sigmoid

classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))

# compiling ANN
classifier.compile(optimizer = 'adam' ,loss = 'binary_crossentropy' , metrics = ['accuracy'])

#fitting the ANN to the trainging set
# batch size is the number of observation after wich we want to update the weights
classifier.fit(X_train , y_train , batch_size = 10 , nb_epoch = 100 )


# part 3 - Making the prediction and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = ( y_pred > 0.5 ) #if y_pred > 0.5 return true
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
