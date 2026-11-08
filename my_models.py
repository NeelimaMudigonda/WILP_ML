# NumPy is used only for maths (arrays, dot products)
import numpy as np

# ================= Scaler (standardization) =================
class MyScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)                 # mean of each column
        self.std_ = X.std(axis=0)                   # standard deviation of each column
        self.std_[self.std_ == 0] = 1               # avoid division by zero
        return self                                 # return object so we can chain calls
    def transform(self, X):
        return (X - self.mean_) / self.std_         # z-score: (x - mean) / std
    def fit_transform(self, X):
        return self.fit(X).transform(X)             # learn mean/std, then apply

# Sigmoid squashes any number into the range 0 to 1 (used as probability)
def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500))) # clip avoids overflow in exp

# ================= 1. Logistic Regression =================
class MyLogisticRegression:
    def __init__(self, lr=0.1, n_iter=2000):
        self.lr, self.n_iter = lr, n_iter           # learning rate, number of iterations
    def fit(self, X, y):
        n, d = X.shape                              # n samples, d features
        self.w, self.b = np.zeros(d), 0.0           # start weights and bias at zero
        for _ in range(self.n_iter):                # repeat gradient descent
            p = _sigmoid(X @ self.w + self.b)       # predicted probability
            self.w -= self.lr * (X.T @ (p - y)) / n # update weights using gradient
            self.b -= self.lr * np.mean(p - y)      # update bias using gradient
        return self
    def predict_proba(self, X):
        return _sigmoid(X @ self.w + self.b)        # probability of class 1
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)  # threshold 0.5

# ================= 2. Decision Tree (Gini impurity) =================
class MyDecisionTree:
    def __init__(self, max_depth=6, min_samples=5):
        self.max_depth, self.min_samples = max_depth, min_samples  # stopping rules
    @staticmethod
    def _gini(y):
        p = y.mean()                                # fraction of class 1
        return 2 * p * (1 - p)                      # Gini impurity for 2 classes
    def _build(self, X, y, depth):
        # stop if too deep, too few samples, or node is already pure
        if depth >= self.max_depth or len(y) < self.min_samples or self._gini(y) == 0:
            return {"leaf": True, "p": y.mean()}    # leaf stores probability of class 1
        best_gain, best = 0, None                   # track best split found
        parent = self._gini(y)                      # impurity before splitting
        for f in range(X.shape[1]):                 # try every feature
            vals = np.unique(X[:, f])               # unique values of this feature
            for t in (vals[:-1] + vals[1:]) / 2:    # candidate thresholds (midpoints)
                left = X[:, f] <= t                 # rows going to left child
                nl, nr = left.sum(), (~left).sum()  # size of left and right
                if nl == 0 or nr == 0:              # skip useless splits
                    continue
                # weighted impurity after split
                child = (nl * self._gini(y[left]) + nr * self._gini(y[~left])) / len(y)
                g = parent - child                  # information gain
                if g > best_gain:                   # keep the best split
                    best_gain, best = g, (f, t, left)
        if best is None:                            # no good split found
            return {"leaf": True, "p": y.mean()}
        f, t, left = best
        return {"leaf": False, "f": f, "t": t,      # internal node: feature + threshold
                "l": self._build(X[left], y[left], depth + 1),    # build left subtree
                "r": self._build(X[~left], y[~left], depth + 1)}  # build right subtree
    def fit(self, X, y):
        self.tree = self._build(X, np.asarray(y), 0)  # build tree from the root
        return self
    def _one(self, x):
        node = self.tree                            # start at root
        while not node["leaf"]:                     # walk down until a leaf
            node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
        return node["p"]                            # probability stored in the leaf
    def predict_proba(self, X):
        return np.array([self._one(x) for x in X])  # probability for every row
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ================= 3. K-Nearest Neighbors =================
class MyKNN:
    def __init__(self, k=7):
        self.k = k                                  # number of neighbours
    def fit(self, X, y):
        self.X_, self.y_ = X, np.asarray(y)         # KNN just memorizes training data
        return self
    def predict_proba(self, X):
        # Euclidean distance from every test row to every training row
        d = np.sqrt(((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))
        idx = np.argsort(d, axis=1)[:, :self.k]     # indices of k closest rows
        return self.y_[idx].mean(axis=1)            # fraction of neighbours that are class 1
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)  # majority vote

# ================= 4. Gaussian Naive Bayes =================
class MyGaussianNB:
    def fit(self, X, y):
        y = np.asarray(y)
        self.stats = {}                             # store mean, variance, prior per class
        for c in (0, 1):
            Xc = X[y == c]                          # rows of class c
            self.stats[c] = (Xc.mean(axis=0),       # mean of each feature
                             Xc.var(axis=0) + 1e-6, # variance (+ tiny value for stability)
                             len(Xc) / len(y))      # prior probability of class c
        return self
    def _log_joint(self, X, c):
        m, v, prior = self.stats[c]
        # sum of log Gaussian likelihoods over all features
        ll = -0.5 * np.sum(np.log(2 * np.pi * v) + (X - m) ** 2 / v, axis=1)
        return ll + np.log(prior)                   # add log prior
    def predict_proba(self, X):
        # difference of log joints -> sigmoid gives P(class 1)
        return _sigmoid(self._log_joint(X, 1) - self._log_joint(X, 0))
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ================= 5. Support Vector Machine (linear, hinge loss) =================
class MySVM:
    def __init__(self, lam=0.01, lr=0.1, n_iter=2000):
        self.lam, self.lr, self.n_iter = lam, lr, n_iter  # regularization, learning rate, iterations
    def fit(self, X, y):
        ys = np.where(np.asarray(y) == 1, 1, -1)    # SVM needs labels -1 and +1
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0           # start at zero
        for i in range(self.n_iter):
            viol = ys * (X @ self.w + self.b) < 1   # points inside the margin or misclassified
            gw = self.lam * self.w - (X[viol].T @ ys[viol]) / n  # gradient for weights
            gb = -ys[viol].sum() / n                # gradient for bias
            step = self.lr / (1 + 0.01 * i)         # learning rate decays over time
            self.w -= step * gw                     # update weights
            self.b -= step * gb                     # update bias
        return self
    def decision_function(self, X):
        return X @ self.w + self.b                  # signed distance from the boundary
    def predict_proba(self, X):
        return _sigmoid(2 * self.decision_function(X))  # convert score to 0-1 (needed for AUC)
    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)  # positive side = class 1

# ================= Metrics from scratch =================
def confusion(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tn = int(((y == 0) & (p == 0)).sum())           # true negatives
    fp = int(((y == 0) & (p == 1)).sum())           # false positives
    fn = int(((y == 1) & (p == 0)).sum())           # false negatives
    tp = int(((y == 1) & (p == 1)).sum())           # true positives
    return tn, fp, fn, tp

def all_metrics(y, pred, prob):
    tn, fp, fn, tp = confusion(y, pred)
    acc = (tp + tn) / (tp + tn + fp + fn)           # Accuracy
    prec = tp / (tp + fp) if tp + fp else 0         # Precision
    rec = tp / (tp + fn) if tp + fn else 0          # Recall
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0  # F1 score
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0   # Matthews correlation coefficient
    y = np.asarray(y)
    pos, neg = prob[y == 1], prob[y == 0]           # scores of positives and negatives
    # AUC = chance a random positive is scored higher than a random negative
    auc = ((pos[:, None] > neg[None, :]).sum()
           + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    return {"Accuracy": acc, "AUC": auc, "Precision": prec,
            "Recall": rec, "F1": f1, "MCC": mcc}
