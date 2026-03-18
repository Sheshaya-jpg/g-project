# class definition of trained model
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:

    def __init__(self, NumTrees=100, LearningRate=0.1, max_depth=3):
        self.NumTrees = NumTrees
        self.LearningRate = LearningRate
        self.max_depth = max_depth
        self.trees = []
        self.InitPredic = None

    def fit(self, X, y):
        self.InitValue = y.mean()
        yPredict = np.full(len(y), self.InitValue)
        for _ in range(self.NumTrees):
            Residuals = y - yPredict
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=3,
                random_state=42,
            )
            tree.fit(X, Residuals)
            yPredict += self.LearningRate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        yPredict = np.full(X.shape[0], self.InitValue)
        for tree in self.trees:
            yPredict += self.LearningRate * tree.predict(X)
        return yPredict