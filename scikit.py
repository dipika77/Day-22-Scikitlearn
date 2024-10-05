#Supervised learning with scikit-learn

#Classification KNN classifier
#instructions
'''Import KNeighborsClassifier from sklearn.neighbors.
Instantiate a KNeighborsClassifier called knn with 6 neighbors.
Fit the classifier to the data using the .fit() method.'''

# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the data
knn.fit(X, y)


#instructions
'''Create y_pred by predicting the target values of the unseen features X_new using the knn model.
Print the predicted labels for the set of predictions.'''

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred)) 


#Measuring model prediction 
#instructions
'''Import train_test_split from sklearn.model_selection.
Split X and y into training and test sets, setting test_size equal to 20%, random_state to 42, and ensuring the target label proportions reflect that of the original dataset.
Fit the knn model to the training data.
Compute and print the model's accuracy for the test data.'''

# Import the module
from sklearn.model_selection import train_test_split

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify= y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test,y_test))


#instructions
'''Create neighbors as a numpy array of values from 1 up to and including 12.
Instantiate a KNeighborsClassifier, with the number of neighbors equal to the neighbor iterator.
Fit the model to the training data.
Calculate accuracy scores for the training set and test set separately using the .score() method, and assign the results to the train_accuracies and test_accuracies dictionaries, respectively, utilizing the neighbor iterator as the index.'''

# Create neighbors
neighbors = np.arange(1,13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors = neighbor)
  
	# Fit the model
	knn.fit(X_train, y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)



# instructions
'''Add a title "KNN: Varying Number of Neighbors".
Plot the .values() method of train_accuracies on the y-axis against neighbors on the x-axis, with a label of "Training Accuracy".
Plot the .values() method of test_accuracies on the y-axis against neighbors on the x-axis, with a label of "Testing Accuracy".
Display the plot.'''

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors,train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()


#INtroduction to regression
#instructions
'''Create X, an array of the values from the sales_df DataFrame's "radio" column.
Create y, an array of the values from the sales_df DataFrame's "sales" column.
Reshape X into a two-dimensional NumPy array.
Print the shape of X and y.'''

import numpy as np

# Create X from the radio column's values
X = sales_df['radio']

# Create y from the sales column's values
y = sales_df['sales']

# Reshape X
X = X.values.reshape(-1,1)

# Check the shape of the features and targets
print(X.shape,y.shape)


#instructions
'''Import LinearRegression.
Instantiate a linear regression model.
Predict sales values using X, storing as predictions.'''

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X,y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])


#instructions
'''Import matplotlib.pyplot as plt.
Create a scatter plot visualizing y against X, with observations in blue.
Draw a red line plot displaying the predictions against X.
Display the plot.'''

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()



#The basics of linear regression
#instructions
'''Create X, an array containing values of all features in sales_df, and y, containing all values from the "sales" column.
Instantiate a linear regression model.
Fit the model to the training data.
Create y_pred, making predictions for sales using the test features.'''

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train,y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))


#instructions
'''Import mean_squared_error.
Calculate the model's R-squared score by passing the test feature values and the test target values to an appropriate method.
Calculate the model's root mean squared error using y_test and y_pred.
Print r_squared and rmse.'''

# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared= False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))


#Cross Validations
#instructions
'''Import KFold and cross_val_score.
Create kf by calling KFold(), setting the number of splits to six, shuffle to True, and setting a seed of 5.
Perform cross-validation using reg on X and y, passing kf to cv.
Print the cv_scores.'''

# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

# Create a KFold object
kf = KFold(n_splits= 6, shuffle= True, random_state= 5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)



#instructions
'''Calculate and print the mean of the results.
Calculate and print the standard deviation of cv_results.
Display the 95% confidence interval for your results using np.quantile().'''

# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))



#Regularized Regression
#instructions
'''mport Ridge.
Instantiate Ridge, setting alpha equal to alpha.
Fit the model to the training data.
Calculate the R-squared score for each iteration of ridge.'''

# Import Ridge
from sklearn.linear_model import Ridge
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
  
  # Create a Ridge regression model
  ridge = Ridge(alpha = alpha)
  
  # Fit the data
  ridge.fit(X_train, y_train)
  
  # Obtain R-squared
  score = ridge.score(X_test, y_test)
  ridge_scores.append(score)
print(ridge_scores)


#instructions
'''mport Lasso from sklearn.linear_model.
Instantiate a Lasso regressor with an alpha of 0.3.
Fit the model to the data.
Compute the model's coefficients, storing as lasso_coef.'''

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regression model
lasso = Lasso(alpha = 0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()