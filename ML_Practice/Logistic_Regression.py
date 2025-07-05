#Train a logistic regression classifier to predict whether a flower is iris virginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int32) # preprocessing (transformed first into T/F then into 0s and 1s)

# Train a logistic regression classifier

clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[2.6]]))
print(example)

#Using matplotlib to plot the visualisation

x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new, y_prob[:,1], "g-", label = "virginica")
plt.show()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])
