# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:22:02 2020

@author: WINDOWS
"""
" LINEAR REGRESSION : It is a way to model the relationship between a dependent and independent variable"
# =============================================================================
# ## importing the neccessary libraries to be used in this algorithm
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
# =============================================================================
# ##Importing Data Set from URL
# =============================================================================
mydata = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print("Data imported successfully")
mydata.head()

# =============================================================================
# # Creating the Independendent and Dependent Data Sets
# =============================================================================
X = mydata.iloc[:, :-1].values #Feature Data
y = mydata.iloc[:, 1].values # Dependent Data
# =============================================================================
# ## Spliting the data to X and Y for training and testing purpose
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# # Fitting Linear Regression to the Training set
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# =============================================================================
# # Predicting the Test set results
# =============================================================================

y_pred = regressor.predict(X_test)
y_pred_data=pd.DataFrame(y_pred)

regressor.score(X_train,y_train)

regressor.score(X_test,y_test)
# =============================================================================
# #Adding Intercept term to the model
# =============================================================================
X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)

# =============================================================================
# #Converting into Dataframe
# =============================================================================
X_train_d=pd.DataFrame(X_train)

# =============================================================================
# #Printing the Model Statistics
# =============================================================================
model = sm.OLS(y_pred,X_test).fit()
model.summary()

# =============================================================================
# ## Plotting the regression line and test data
# =============================================================================

line = regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line,c='g')
x= mydata['Hours']
y= mydata['Scores']
plt.plot(x,y,'*',c='r')
plt.title("TASK 2 : Supervised Learning")
plt.xlabel("Hours of studying")
plt.ylabel("Percentage of marks")
plt.show

# =============================================================================
##Testing with own data
# =============================================================================
hours = 9
own_pred = regressor.predict(np.array(hours).reshape(-1,1))
print("  No of Hours = {}".format(hours))
print("  Predicted Score = {}".format(own_pred[0]))

# =============================================================================
# # Validation using R2 Score 
# =============================================================================
from sklearn.metrics import r2_score
print(" The R2 score is : ", r2_score(y_test,y_pred))