# TF-IDF + Logistic Regression/SVM baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class TfidfLogRegBaseline:
    def __init__(self, model_type='logreg'):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        if model_type == 'logreg':
            self.model = LogisticRegression(max_iter=200)
        else:
            self.model = SVC(probability=True)
    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    def predict_proba(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)
