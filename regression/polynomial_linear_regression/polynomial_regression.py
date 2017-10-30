import pandas as pd
import numpy as np
#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values #make sure that X is matrix and not a vector
y = dataset.iloc[:,2].values
#print(X)

#Simple linear to compare the results
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4 )
X_poly = poly_reg.fit_transform(X)
print(X_poly)

#Create new linear regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly , y)

# Visualising the Linear regression results
import matplotlib.pyplot as plt
plt.scatter(X , y,color='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear regression) ')
plt.xlabel('Position level')
plt.ylabel('Salary')
#plt.show()

# Visualising the Polynomial regression results
import matplotlib.pyplot as plt

#for high res
X_grid = np.arange(min(X),max(X) , 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X , y,color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial regression) ')
plt.xlabel('Position level')
plt.ylabel('Salary')
#plt.show()

#Predicting a new result with linear regression
#instead of passing a matrix to func , simply pass a value
print(lin_reg.predict(6.5))


#Predicting a new result with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))
