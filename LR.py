import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
multiplicities = np.load('data@200GeV.npy')

# Create transverse momenta array based on the array indices
transverse_momenta = np.arange(len(multiplicities))

# Transform the output variable using the natural logarithm
log_multiplicities = np.log(multiplicities + 1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(transverse_momenta.reshape(-1, 1), log_multiplicities, test_size=0.2, random_state=42)

# Create sample weights based on the original multiplicity values
sample_weights = np.array([multiplicities[i] for i in X_train.ravel()])

# Create and train the linear regression model with sample weights
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate the model
y_pred_train = linear_regression.predict(X_train)
y_pred_test = linear_regression.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train, sample_weight=sample_weights)
test_mse = mean_squared_error(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train, sample_weight=sample_weights)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Train MSE: {train_mse}, Train MAE: {train_mae}")
print(f"Test MSE: {test_mse}, Test MAE: {test_mae}")

# Define the target function
def target_function(transverse_momentum):
    log_prediction = linear_regression.predict(np.array([transverse_momentum]).reshape(-1, 1))
    return np.exp(log_prediction)[0]

# Visualize the target function
transverse_momentum_values = np.arange(len(multiplicities))
predicted_multiplicities = [target_function(t) for t in transverse_momentum_values]

plt.figure(figsize=(10, 6))
plt.scatter(transverse_momenta, multiplicities, color='blue', alpha=0.5, label='Data')
plt.plot(transverse_momentum_values, predicted_multiplicities, color='red', label='Target function')
plt.xlabel('Transverse Momentum')
plt.ylabel('Multiplicity')
plt.title('Relation between Transverse Momentum and Multiplicity (Log-Linear Regression with Sample Weights)')
plt.legend()
plt.show()

