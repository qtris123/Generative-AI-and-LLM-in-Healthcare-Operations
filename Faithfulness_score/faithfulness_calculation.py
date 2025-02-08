### Import pre-trained models and explainers to calculate the Faithfulness score of the explainers ###
import numpy as np
import torch
import pandas as pd
import pickle

from explainer import Explainer
from perturbation_method import NormalPerturbation

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter("ignore", category=UserWarning)

def conv_disc_inds_to_char_enc(discrete_feature_indices: list[int], n_features: int):
    """Converts an array of discrete feature indices to a char encoding.

    Here, the ith value in the returned array is 'c' or 'd' for whether the feature is
    continuous or discrete respectively.

    Args:
        discrete_feature_indices: An array like [0, 1, 2] where the ith value corresponds to
                                  whether the arr[i] column in the data is discrete.
        n_features: The number of features in the data.
    Returns:
        char_encoding: An encoding like ['c', 'd', 'c'] where the ith value indicates whether
                       that respective column in the data is continuous ('c') or discrete ('d')
    """
    # Check to make sure (1) feature indices are integers and (2) they are unique
    error_message = "Features all must be type int but are not"
    assert all(isinstance(f, int) for f in discrete_feature_indices), error_message
    error_message = "Features indices must be unique but there are repetitions"
    assert len(set(discrete_feature_indices)) == len(discrete_feature_indices), error_message
    # Perform conversion
    char_encoding = ['e'] * n_features
    for i in range(len(char_encoding)):
        if i in discrete_feature_indices:
            char_encoding[i] = 'd'
        else:
            char_encoding[i] = 'c'
    # In case something still went wrong
    assert 'e' not in char_encoding, 'Error in char encoding processing!'
    return char_encoding

class FaithfulnessScore:
    def __init__(self, mode='SHAP'):
        if mode == 'LIME':
            explainer = Explainer(mode)
            self.lime_explainer = explainer.explainer_model #LIME explainer
        elif mode =='SHAP':
            explainer = Explainer(mode)
            self.not_trained = explainer.not_trained #during the first attempt (when not_trained == True), a RF model and its SHAP explanation for X_train data should alr be saved. Hence, no need to run the SHAP explainer again.
            self.shap_explainer = explainer.explainer_model # SHAP explainer  

        self.mode = mode
        self.data = explainer.cleaned_data #cleaned, full data, pd.Dataframe
        self.model = explainer.model #model
        self.labels = explainer.labels #labels
        self.test_data = explainer.test_data #test_data

        self.discrete_features = [2,3,4,5,6,7] # I extract these numbers manually from model.feature_names

         # TODO(satya): change this to be inputs to __init__
        # The criteria used to perturb the explanation point and determine which explanations
        # are the most faithful
        self.perturbation_mean = 0.0
        self.perturbation_std = 0.05
        self.perturbation_flip_percentage = 0.03
        self.perturbation_max_distance = 0.4

        # This is a bit clearer, instead of making users use this representation + is the way
        # existing explanation packages (e.g., LIME do it.)
        self.feature_types = conv_disc_inds_to_char_enc(discrete_feature_indices=self.discrete_features,
                                                        n_features=self.data.shape[1]-1) # -1 to exclude the target column

        self.perturbation_method = NormalPerturbation("tabular",
                                                      mean=self.perturbation_mean,
                                                      std=self.perturbation_std,
                                                      flip_percentage=self.perturbation_flip_percentage)
    


    def _compute_faithfulness_auc(self, metric="topk", k = 3):
        """Computes AUC for faithfulness scores, perturbing top k (where k is an array).

        Args:
            k: the top k features to perturb
            metric:
        Returns:
            faithfulness:
        """
        faithfulness = 0
        # iterate through the impact from top-1 to top-k feature
        for k_i in range(1,k+1):
            if self.mode == "LIME":
                # select explanation values for one instance/row; at the moment, take 9 as an example.
                exp_instance = self.lime_explainer.explain_instance(np.array(self.test_data[self.test_data.index == 9])[0], self.model.predict_proba, num_features=14, top_labels=k_i)
                c_label = list(exp_instance.predict_proba).index(max(exp_instance.predict_proba)) #the most likely class
                feature_contribution_scores = exp_instance.as_map()[c_label]

                # Construct original mask as all true (i.e., all indices are masked and non are perturbed)
                top_k_map = torch.tensor([True] * len(feature_contribution_scores), dtype=torch.bool)

                # Unmask topk instances 
                top_k_indices = [ x[0] for x in exp_instance.as_map()[c_label]][:k_i]
                top_k_map[top_k_indices] = False


            elif self.mode == "SHAP":
                if (self.not_trained):
                    explanation = self.shap_explainer(self.test_data)
                    with open('shap_explanation.pkl', 'wb') as file:
                        pickle.dump(explanation, file)
                else:
                    with open('shap_explanation.pkl', 'rb') as file:
                        explanation = pickle.load(file)
                # select explanation values for one instance/row; at the moment, take 9 as an example
                X_test_copy = self.test_data.copy()
                X_test_copy['index'] = range(X_test_copy.shape[0])
                ind = X_test_copy[X_test_copy.index == 9]['index'].values[0] #add a (sorted) index column to hep find the index of the row for the explaination values.
                # the final shap value = base value + sum of the feature contributions (for each class)
                sum_features = np.sum(explanation[ind,:,:].values, axis = 0) # dim(n_features x n_classes)
                base_values = self.shap_explainer.expected_value # dim(1 x n_classes) #use shap_explainer not explanation
                c_label = np.argmax(base_values + sum_features) # the most likely class for an instance
                feature_contribution_scores = explanation[ind,:,c_label]

                # Construct original mask as all true (i.e., all indices are masked and non are perturbed)
                top_k_map = torch.tensor([True] * len(feature_contribution_scores), dtype=torch.bool)

                # Unmask topk instances 
                top_k_indices = list(np.argsort(abs(feature_contribution_scores.values))[-k_i:])
                top_k_map[top_k_indices] = False

            # If top-k provide top-k instances 
            x = np.array(self.test_data[self.test_data.index == 9]) # x input for compute_faithfulness_topk is an np.array with reshape(1,-1) or without [0]

                
            if metric == "topk":
                faithfulness += self._compute_faithfulness_topk(x, c_label, top_k_map)
            else:
                # Otherwise, provide bottom-k indices
                faithfulness += self._compute_faithfulness_topk(x, c_label, ~top_k_map)
        return faithfulness
    

    def _compute_faithfulness_topk(self, x, label, top_k_mask, num_samples: int = 10_000):
        """Approximates the expected local faithfulness of the explanation in a neighborhood.

        Args:
            x: The original sample
            label: the most likely class
            top_k_mask:
            num_samples: number of perturbations used for Monte Carlo expectation estimate
        """
        perturb_args = {
            "original_sample": x[0],
            "feature_mask": top_k_mask,
            "num_samples": num_samples,
            "max_distance": self.perturbation_max_distance,
            "feature_metadata": self.feature_types
        }
        # Compute perturbed instance
        x_perturbed = self.perturbation_method.get_perturbed_inputs(**perturb_args)
        # TODO(satya): Could you make these lines more readable?
        y = self._arr([i[label] for i in self._arr(self.model.predict_proba(x.reshape(1, -1)))])
        y_perturbed = self._arr([i[label] for i in self._arr(self.model.predict_proba(x_perturbed.float()))])

        # Return abs mean
        return np.mean(np.abs(y - y_perturbed), axis=0)
    
    
    @staticmethod
    def _arr(x) -> np.ndarray:
        """Converts x to a numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    """""
    def explain_instance(self,
                         data: Union[np.ndarray, pd.DataFrame],
                         top_k_starting_pct: float = 0.2,
                         top_k_ending_pct: float = 0.5,
                         epsilon: float = 1e-4,
                         return_fidelities: bool = False) -> MegaExplanation:
        if not isinstance(data, np.ndarray):
            try:
                data = data.to_numpy()
            except Exception as exp:
                message = f"Data not type np.ndarray, failed to convert with error {exp}"
                raise NameError(message)

        explanations, scores = {}, {}
        fidelity_scores_topk = {}

        # Makes sure data is formatted correctly
        formatted_data = self.format_data(data)

        # Gets indices of 20-50% of data
        lower_index = int(formatted_data.shape[1]*top_k_starting_pct)
        upper_index = int(formatted_data.shape[1]*top_k_ending_pct)
        k = list(range(lower_index, upper_index))

        # Explain the most likely class
        label = np.argmax(self.model(formatted_data)[0])

        # Iterate over each explanation method and compute fidelity scores of topk
        # and non-topk features per the method
        for method in self.explanation_methods.keys():
            cur_explainer = self.explanation_methods[method]
            cur_expl, score = cur_explainer.get_explanation(formatted_data,
                                                            label=label)

            explanations[method] = cur_expl.squeeze(0)
            scores[method] = score
            # Compute the fidelity auc of the top-k features
            fidelity_scores_topk[method] = self._compute_faithfulness_auc(formatted_data,
                                                                          explanations[method],
                                                                          label,
                                                                          k,
                                                                          metric="topk")
    

    def format_data(data_x: pd.DataFrame) -> np.ndarray:
        ""Checks to make sure the data being explained is a single instance and 1-dim.""
        # Check to make sure data_x is an individual sample
        first_row_array = data_x.iloc[0].to_numpy()
        return first_row_array.reshape(1,-1)
    """""

obj = FaithfulnessScore()
print(f'the faithful score is: {obj._compute_faithfulness_auc()}')
