import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

iris = datasets.load_diabetes()
for i in range(0,10):
    iris_X = iris.data[:, np.newaxis, i]
    iris_X_train = iris_X[:-30]
    iris_X_test = iris_X[-30:]

    iris_y_train = iris.target[:-30]
    iris_y_test = iris.target[-30:]

    model = linear_model.LinearRegression()
    model.fit(iris_X_train, iris_y_train)
    iris_y_predicted = model.predict(iris_X_test)

    print("Mean squared error is: ", mean_squared_error(iris_y_predicted, iris_y_test))
    print("Weights", model.coef_)
    print("Intercept", model.intercept_)
    plt.scatter(iris_X_test, iris_y_test, color='black', label='Actual')
    plt.plot(iris_X_test, iris_y_predicted, color='blue', linewidth=2, label='Predicted')
    plt.legend()
    plt.show()