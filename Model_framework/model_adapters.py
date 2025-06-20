from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
from base_adapter import BaseModelAdapter
from gridsearch_mixin import GridSearchMixin
from shap_mixin import SHAPMixin
from lime_mixin import LIMEMixin

class RandomForestAdapter(BaseModelAdapter, GridSearchMixin, SHAPMixin, LIMEMixin):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def get_model(self):
        return self.model

class LogisticRegressionAdapter(BaseModelAdapter, GridSearchMixin, SHAPMixin, LIMEMixin):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def get_model(self):
        return self.model

class MLPClassifierAdapter(BaseModelAdapter, GridSearchMixin, SHAPMixin, LIMEMixin):
    def __init__(self, **kwargs):
        self.model = MLPClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def get_model(self):
        return self.model

class SVMAdapter(BaseModelAdapter, GridSearchMixin, SHAPMixin, LIMEMixin):
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def get_model(self):
        return self.model

class AdaBoostAdapter(BaseModelAdapter, GridSearchMixin, SHAPMixin, LIMEMixin):
    def __init__(self, **kwargs):
        self.model = AdaBoostClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def get_model(self):
        return self.model

class GradientBoostingAdapter(BaseModelAdapter, GridSearchMixin, SHAPMixin, LIMEMixin):
    def __init__(self, **kwargs):
        self.model = GradientBoostingClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def get_model(self):
        return self.model 