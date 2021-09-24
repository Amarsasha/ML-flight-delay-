import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
import datetime
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


################# Load and Explore the dataset
flight_delay_df = pd.read_csv('flight_delay.csv')

# Encoding
ae = preprocessing.LabelEncoder()
flight_delay_df['Depature Airport'] = ae.fit_transform(flight_delay_df['Depature Airport'])
flight_delay_df['Destination Airport'] = ae.fit_transform(flight_delay_df['Destination Airport'])

# extracting time features
flight_delay_df['departure_year'] = pd.DatetimeIndex(flight_delay_df['Scheduled depature time']).year
flight_delay_df['arrival_year'] = pd.DatetimeIndex(flight_delay_df['Scheduled arrival time']).year
flight_delay_df['departure_dayofyear'] = pd.DatetimeIndex(flight_delay_df['Scheduled depature time']).dayofyear
flight_delay_df['arrival_dayofyear'] = pd.DatetimeIndex(flight_delay_df['Scheduled arrival time']).dayofyear

# Adding a new feature - flight duration
duration = []
for i in range(len(flight_delay_df['Scheduled arrival time'])):
    duration.append((datetime.datetime.strptime(str(flight_delay_df['Scheduled arrival time'][i]),"%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(str(flight_delay_df['Scheduled depature time'][i]),"%Y-%m-%d %H:%M:%S")).total_seconds())
flight_delay_df['Flight duration'] = duration

# ## Data Statistics
flight_delay_df.describe()

# ## Visualization
flight_delay_df.plot(x='Flight duration', y='Delay', style='o')
plt.title('Flight duration vs Delay')
plt.xlabel('Duration')
plt.ylabel('Delay')
plt.show()


### Preprocess the Data & split to train set and test set

#flight_delay_df = flight_delay_df.drop(flight_delay_df[flight_delay_df['Flight duration'] > 50000].index)

X = flight_delay_df.drop(['Delay','Scheduled depature time', 'Scheduled arrival time'], axis = 1)
y = flight_delay_df['Delay']

X_train = X.drop(flight_delay_df[flight_delay_df['departure_year'] == 2018].index)
y_train = y.drop(flight_delay_df[flight_delay_df['departure_year'] == 2018].index)
X_test = X.drop(flight_delay_df[flight_delay_df['departure_year'] != 2018].index)
y_test = y.drop(flight_delay_df[flight_delay_df['departure_year'] != 2018].index)

# ## Outlier Detection & Removal
sns.boxplot(data=flight_delay_df,x=flight_delay_df["Flight duration"])
plt.title("Boxplot of Swiss Banknote Length ")

y_train = y_train.drop(X_train[X_train['Flight duration'] > 50000].index)
X_train = X_train.drop(X_train[X_train['Flight duration'] > 50000].index)

flight_delay_df.plot(x='Flight duration', y='Delay', style='o')

plt.title('Flight duration vs Delay')
plt.xlabel('Duration')
plt.ylabel('Delay')
plt.show()

####################### Machine learning models

####################### 1 Linear regression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(f"Model intercept : {regressor.intercept_}")
print(f"Model coefficient : {regressor.coef_}")

y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

print("\nLinear regression")
print("The model performance for the training set")
print("-------------------------------------------")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred_train))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred_train))
print('R squared', metrics.r2_score(y_train, y_pred_train))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))

print("\n")

print("The model performance for the test set")
print("-------------------------------------------")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_test))
print('R squared', metrics.r2_score(y_test, y_pred_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))

######################## 2 Polynomial regression
from sklearn.preprocessing import PolynomialFeatures #to convert the original features into their higher order terms 
from sklearn.linear_model import LinearRegression

def create_polynomial_regression_model(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
  
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_train_predict = poly_model.predict(X_train_poly)
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
    
    # printing of metrics
    print("\nPolynomial regression")
    print("The model performance for the training set")
    print("-------------------------------------------")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_train_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_train_predict))
    print('R squared', metrics.r2_score(y_train, y_train_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
    
    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")   
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_predict))
    print('R squared', metrics.r2_score(y_test, y_test_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

create_polynomial_regression_model(2)

########################## 3 Lasso regularization
X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=123)
alphas = [2.2, 2, 1.5, 1.3, 1.2, 1.1, 1, 0.3, 0.1]
losses = []
 
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)
    losses.append(mse)
plt.plot(alphas, losses)
plt.title("Lasso alpha value selection")
plt.xlabel("alpha")
plt.ylabel("Mean squared error")
plt.show()
 
best_alpha = alphas[np.argmin(losses)]
print("Best value of alpha:", best_alpha)

lasso = Lasso(best_alpha)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print("\nLasso regularization")
print("The model performance for the training set")
print("-------------------------------------------")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_train_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_train_pred))
print('R squared', metrics.r2_score(y_train, y_train_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

print("\n")

print("The model performance for the test set")
print("-------------------------------------------")   
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('R squared', metrics.r2_score(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))