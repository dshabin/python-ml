import pandas as pd
import numpy as np
#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4]
#print(X)

#Encoding variables
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
#print(X)

#create dummy variables
from sklearn.preprocessing import OneHotEncoder
#which colum should be encoded
onehotencoder = OneHotEncoder(categorical_features = [3]) # 3 is the index of column for hot encode
X = onehotencoder.fit_transform(X).toarray()
#print(X)

#avoiding the dummy variable trap
X = X[:,1:]


#spliting data set
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state = 0 )
#print(X_test)

#firtting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

#predicting the Test set result
y_pred = regressor.predict(X_test)
#print(y_pred)
#print(y_test)


#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
# add columun of ones to the futre matrix of X
#X = np.append(arr = X , values = np.ones((50,1)).astype(int) ,  axis = 1 ) # axis = 1 is column

# change the X and values for adding the columns of one to the begining of X
X = np.append(arr = np.ones((50,1)).astype(int) , values = X  ,  axis = 1 )
#print(X)

#Building backward elimination
X_opt = X[:,[0,1,2,3,4,5]] #intialize with all data
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() # Backward elimination step 2
# get P value for each independent value and check it
print(regressor_OLS.summary())
#remove the x2 , beacause ita has the highest p value
X_opt = X[:,[0,1,3,4,5]] #backward elimination step 4
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() # Backward elimination step 5
# get P value for each independent value and check it
print(regressor_OLS.summary())

#repeat
X_opt = X[:,[0,3,4,5]] #backward elimination step 4
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() # Backward elimination step 5
# get P value for each independent value and check it
print(regressor_OLS.summary())

#repeat
X_opt = X[:,[0,3,5]] #backward elimination step 4
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() # Backward elimination step 5
# get P value for each independent value and check it
print(regressor_OLS.summary())

#repeat
X_opt = X[:,[0,3]] #backward elimination step 4
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() # Backward elimination step 5
# get P value for each independent value and check it
print(regressor_OLS.summary())
