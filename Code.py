import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
# Read the data from CSV
data = pd.read_csv('data.csv')
X = data[['Feature 1', 'Feature 2']]
y = data['Target']
# Perform Ridge regression
ridge = Ridge(alpha=1.0)  # You can adjust the regularization strength with the alpha parameter
ridge.fit(X, y)
# Visualize the results
predictions = ridge.predict(X)
plt.scatter(y, predictions)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Ridge Regression Results')
plt.show()


from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
# Perform Lasso Bayesian analysis
lasso_bayesian = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
lasso_bayesian.fit(X, y)
# Visualize the results
predictions = lasso_bayesian.predict(X)
plt.scatter(y, predictions)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Lasso Bayesian Results')
plt.show()
