import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import List, Tuple

class LIMEMixin:
    def lime_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame, num_features: int = 10, num_samples: int = 5) -> List[Tuple]:
        """
        Calculate LIME explanations for a sample of test instances.
        
        Args:
            X_train: Training data used to create the explainer
            X_test: Test data to explain
            num_features: Number of features to show in explanation
            num_samples: Number of test instances to explain
            
        Returns:
            List of tuples containing (instance, explanation) pairs
        """
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=['0', '1', '2'],  # Assuming 3 classes for placement
            mode='classification'
        )
        
        # Select random samples from test set
        sample_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_indices]
        
        explanations = []
        for idx in sample_indices:
            # Get explanation for this instance
            exp = explainer.explain_instance(
                X_test.iloc[idx].values,
                self.model.predict_proba,
                num_features=num_features
            )
            explanations.append((X_test.iloc[idx], exp))
            
        return explanations 