import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

def euc(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def man(x, y):
    return np.sum(np.abs(x - y))

def cos(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

dist_map = {
    'euclidean': euc,
    'manhattan': man,
    'cosine': cos
}

class knn:
    def __init__(self, k=3, dist='euclidean'):
        self.k = k
        self.distfunc = dist_map[dist]
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = [self.distfunc(x, x2) for x2 in self.X]
            idx = np.argsort(dists)[:self.k]
            lab = [self.y[i] for i in idx]
            vote = Counter(lab).most_common(1)[0][0]
            preds.append(vote)
        return preds

def run_model(X, y, ks=[1,3,5,7,10], dists=['euclidean', 'manhattan', 'cosine'], folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for d in dists:
        print("\nDist:", d)
        for k in ks + [len(X)-1]:
            print("K =", k)
            ytrue = []
            ypred = []
            for train_idx, test_idx in kf.split(X):
                xtr = X[train_idx]
                xte = X[test_idx]
                ytr = y[train_idx]
                yte = y[test_idx]
                model = knn(k=k, dist=d)
                model.fit(xtr, ytr)
                p = model.predict(xte)
                ytrue.extend(yte)
                ypred.extend(p)
            print("CM:")
            print(confusion_matrix(ytrue, ypred))
            print("Report:")
            print(classification_report(ytrue, ypred, zero_division=0))

data = load_iris()
X = data.data
y = data.target
run_model(X, y)
