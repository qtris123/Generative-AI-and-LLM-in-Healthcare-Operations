import shap
import pandas as pd

class SHAPMixin:
    def shap_values(self, X_train : pd.DataFrame, X_test : pd.DataFrame, explainer_type : str):
        if explainer_type in ['tree', 'adaboost', 'gb']:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer(X_test)
            return shap_values
        elif explainer_type == 'logistic':
            explainer = shap.LinearExplainer(self.model, masker=shap.maskers.Independent(X_train))
            shap_values = explainer(X_test)
            return shap_values
        elif explainer_type == 'mlp':
            explainer = shap.KernelExplainer(self.model.predict, X_train.sample(25, random_state=50))
            shap_values = explainer(X_test)
            return shap_values
        elif explainer_type == 'svm':
            explainer = shap.KernelExplainer(self.model.predict_proba, X_train)
            shap_values = explainer.shap_values(X_test)
            return shap_values
        else:
            raise ValueError(f"Invalid explainer type: {explainer_type}")

