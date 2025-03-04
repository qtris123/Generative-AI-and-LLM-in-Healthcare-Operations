### Import pre-trained models and explainers to calculate the Faithfulness score of the explainers ###
import numpy as np
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from collections import Counter

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
        self.feature_names = explainer.feature_names_dict #feature names in order of index -> dict{index: name}

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
    

    # the top 5 most frequent features and their appearances 
    def _top_k_features_of_row(self, row_ind=9, k =5, num_cycles = 50):
        """Computes the faithfulness of a single row.

        Args:
            row_ind: the index of the row to collect the top k features for
        Returns:
            the top 5 most frequent features and their appearances
        """
        #faithfulness = 0
        if row_ind == None:
            row_ind = self.test_data.sample(n= 1, random_state=40).index
        
        top_k_features_row = {}

        if self.mode == "LIME":
            # repeat the calcuation for 50 cycles (for 1 row)
            for cycle in tqdm(range(1, num_cycles+1)):
                exp_instance = self.lime_explainer.explain_instance(np.array(self.test_data[self.test_data.index == row_ind])[0], self.model.predict_proba, num_features=14, top_labels=3)
                c_label = list(exp_instance.predict_proba).index(max(exp_instance.predict_proba))
                feature_contribution_scores = exp_instance.as_map()[c_label]

                for k_i in range(1,k+1):
                    # Construct original mask as all true (i.e., all indices are masked and non are perturbed)
                    top_k_map = torch.tensor([True] * len(feature_contribution_scores), dtype=torch.bool)

                    # Unmask topk instances 
                    top_k_indices = [x[0] for x in exp_instance.as_map()[c_label]][:k_i]
                    top_k_map[top_k_indices] = False
                    #self.top_k_features += Counter({self.feature_names[i]: 1 for i, flag in enumerate(top_k_map) if not flag.item()}

                    # If top-k provide top-k instances 
                    x = np.array(self.test_data[self.test_data.index == row_ind]) # x input for compute_faithfulness_topk is an np.array with reshape(1,-1) or without [0]
                    #faithfulness += self._compute_faithfulness_topk(x, c_label, top_k_map)

                #count the occurence of feature in top k in this format: {‘feature_name’: {rank_1: 1 (times),rank_ 2: 3 (times),rank_ 3: 10 (times)}, 'feature_name_2': ... }
                for rank, indx in enumerate(top_k_indices, start=1):
                    feature_name  = self.feature_names[indx]
                    if feature_name not in top_k_features_row:
                        top_k_features_row[feature_name] = {}

                    rank_key = f"rank_{rank}"
                    # Increment the count for this rank or set it to 1 if it doesn't exist yet
                    top_k_features_row[feature_name][rank_key] = top_k_features_row[feature_name].get(rank_key, 0) + 1

        elif self.mode == "SHAP":
            if (self.not_trained):
                explanation = self.shap_explainer(self.test_data)
                with open('shap_explanation.pkl', 'wb') as file:
                    pickle.dump(explanation, file)
            else:
                with open('shap_explanation.pkl', 'rb') as file:
                    explanation = pickle.load(file)
            # repeat the calcuation for 50 cycles (for 1 row)
            for cycle in tqdm(range(1, num_cycles+1)):
                X_test_copy = self.test_data.copy()
                X_test_copy['index'] = range(X_test_copy.shape[0])
                ind = X_test_copy[X_test_copy.index == row_ind]['index'].values[0] #add a (sorted) index column to help find the index of the row for the explaination values.
                # the final shap value = base value + sum of the feature contributions (for each class)
                sum_features = np.sum(explanation[ind,:,:].values, axis = 0) # dim(n_features x n_classes)
                base_values = self.shap_explainer.expected_value # dim(1 x n_classes) #use shap_explainer not explanation
                c_label = np.argmax(base_values + sum_features) # the most likely class for an instance
                feature_contribution_scores = explanation[ind,:,c_label]

                for k_i in range(1,k+1):
                    # Construct original mask as all true (i.e., all indices are masked and non are perturbed)
                    top_k_map = torch.tensor([True] * len(feature_contribution_scores), dtype=torch.bool)
                    # Unmask topk instances 
                    top_k_indices = list(np.argsort(abs(feature_contribution_scores.values))[-k_i:])
                    top_k_map[top_k_indices] = False

                    # If top-k provide top-k instances 
                    x = np.array(self.test_data[self.test_data.index == row_ind]) # x input for compute_faithfulness_topk is an np.array with reshape(1,-1) or without [0]
                    #faithfulness += self._compute_faithfulness_topk(x, c_label, top_k_map)
                
                #count the occurence of feature in top k in this format: {‘feature_name’: {rank_1: 1 (times),rank_ 2: 3 (times),rank_ 3: 10 (times)}, 'feature_name_2': ... }
                for rank, indx in enumerate(top_k_indices, start=1):
                    feature_name  = self.feature_names[indx]
                    if feature_name not in top_k_features_row:
                        top_k_features_row[feature_name] = {}

                    rank_key = f"rank_{rank}"
                    # Increment the count for this rank or set it to 1 if it doesn't exist yet
                    top_k_features_row[feature_name][rank_key] = top_k_features_row[feature_name].get(rank_key, 0) + 1


        """"
        all_N_map = torch.tensor([False] * len(feature_contribution_scores), dtype=torch.bool)
        baseline_score = k*self._compute_faithfulness_topk(x, c_label, all_N_map)
        avg_faith_score = ((faithfulness / num_cycles) / baseline_score)*100
        """

        aggregated_counts = {}
        for feature, ranks in top_k_features_row.items():
            aggregated_counts[feature] = sum(ranks.values())

        top_k_keys = sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_features_sorted = {k: aggregated_counts[k] for k, v in top_k_keys}

        print(aggregated_counts)

        return row_ind, top_k_features_sorted
    

    # calculate the faithfulness score considering an n sample of the test data, and take the average. => the representative faith score for the data
    def _compute_faithfulness_auc(self, metric="topk", k = 5, num_cycles = 1000, num_rows = 10): 
        """Computes AUC for faithfulness scores, perturbing top k (where k is an array).

        Args:
            k: the top k features to perturb
            metric:
        Returns:
            faithfulness:
        """
        top_k_features = {}
        ans = []

        right_distr = []
        left_distr = []

        for i in range(1,num_cycles + 1):
            faithfulness = 0
            # iterate through the impact from top-1 to top-k feature
            indices = self.test_data.sample(n= num_rows).index #, random_state=40).index
            for row_ind in tqdm(indices):
                if self.mode == "LIME":
                    # select explanation values for one instance/row; at the moment, take row indexed 9 as an example.
                    exp_instance = self.lime_explainer.explain_instance(np.array(self.test_data[self.test_data.index == row_ind])[0], self.model.predict_proba, num_features=14, top_labels=k)
                    c_label = list(exp_instance.predict_proba).index(max(exp_instance.predict_proba)) #the most likely class
                    feature_contribution_scores = exp_instance.as_map()[c_label]
                    for k_i in range(1,k+1):
                        # Construct original mask as all true (i.e., all indices are masked and non are perturbed)
                        top_k_map = torch.tensor([True] * len(feature_contribution_scores), dtype=torch.bool)

                        # Unmask topk instances 
                        top_k_indices = [x[0] for x in exp_instance.as_map()[c_label]][:k_i]
                        top_k_map[top_k_indices] = False
                        #self.top_k_features += Counter({self.feature_names[i]: 1 for i, flag in enumerate(top_k_map) if not flag.item()}

                        # If top-k provide top-k instances 
                        x = np.array(self.test_data[self.test_data.index == row_ind]) # x input for compute_faithfulness_topk is an np.array with reshape(1,-1) or without [0]
                        if metric == "topk":
                            faithfulness += self._compute_faithfulness_topk(x, c_label, top_k_map)
                        else:
                            # Otherwise, provide bottom-k indices
                            faithfulness += self._compute_faithfulness_topk(x, c_label, ~top_k_map)

                elif self.mode == "SHAP":
                    if (self.not_trained):
                        explanation = self.shap_explainer(self.test_data)
                        with open('shap_explanation.pkl', 'wb') as file:
                            pickle.dump(explanation, file)
                    else:
                        with open('shap_explanation.pkl', 'rb') as file:
                            explanation = pickle.load(file)
                    # select explanation values for one instance/row; at the moment, take row indexed 9 as an example
                    X_test_copy = self.test_data.copy()
                    X_test_copy['index'] = range(X_test_copy.shape[0])
                    ind = X_test_copy[X_test_copy.index == row_ind]['index'].values[0] #add a (sorted) index column to help find the index of the row for the explaination values.
                    # the final shap value = base value + sum of the feature contributions (for each class)
                    sum_features = np.sum(explanation[ind,:,:].values, axis = 0) # dim(n_features x n_classes)
                    base_values = self.shap_explainer.expected_value # dim(1 x n_classes) #use shap_explainer not explanation
                    c_label = np.argmax(base_values + sum_features) # the most likely class for an instance
                    feature_contribution_scores = explanation[ind,:,c_label]

                    for k_i in range(1,k+1):
                        # Construct original mask as all true (i.e., all indices are masked and non are perturbed)
                        top_k_map = torch.tensor([True] * len(feature_contribution_scores), dtype=torch.bool)
                        # Unmask topk instances 
                        top_k_indices = list(np.argsort(abs(feature_contribution_scores.values))[-k_i:])
                        top_k_map[top_k_indices] = False

                        # If top-k provide top-k instances 
                        x = np.array(self.test_data[self.test_data.index == row_ind]) # x input for compute_faithfulness_topk is an np.array with reshape(1,-1) or without [0]
                        if metric == "topk":
                            faithfulness += self._compute_faithfulness_topk(x, c_label, top_k_map)
                        else:
                            # Otherwise, provide bottom-k indices
                            faithfulness += self._compute_faithfulness_topk(x, c_label, ~top_k_map)


                #count the occurence of feature in top k in this format: {‘feature_name’: {rank_1: 1 (times),rank_ 2: 3 (times),rank_ 3: 10 (times)}, 'feature_name_2': ... }
                for rank, indx in enumerate(top_k_indices, start=1):
                    feature_name  = self.feature_names[indx]
                    if feature_name not in top_k_features:
                        top_k_features[feature_name] = {}

                    rank_key = f"rank_{rank}"
                    # Increment the count for this rank or set it to 1 if it doesn't exist yet
                    top_k_features[feature_name][rank_key] = top_k_features[feature_name].get(rank_key, 0) + 1


            #calculate the baseline:
            all_N_map = torch.tensor([False] * len(feature_contribution_scores), dtype=torch.bool)
            baseline_score = k*self._compute_faithfulness_topk(x, c_label, all_N_map)
            faith_score = ((faithfulness / num_rows) / baseline_score)*100
            ans.append(faith_score)

            
            # 29.225 mean value taken a distribution of 1000 samplings, each representing 10 rows for SHAP; 24.362 for LIME
            faith_avg = 29.225 if self.model == "SHAP" else 24.362
            if faith_score > faith_avg:    
                right_distr.append(indices)
            else:
                left_distr.append(indices)

        

        return ans, top_k_features
    

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
    
    @staticmethod
    def _component_percentage(dict) -> dict:
        """ Returns the percentage of a feature in a specific ranking.
            dict: the dictionary containing the feature counts
            rank_no: the rank number to check | format: 'rank_1', 'rank_2', etc.
            feature: the feature to check | format: 'num_previous_recidivisms', 'age', etc.
        """
        component_percentage = {}
        for feature, value in dict.items():
            for rank, occurence in dict[feature].items():
                sum = 0
                for key, value in dict.items():
                    sum += dict[key].get(rank, 0)
                
                if feature not in component_percentage.keys():
                    component_percentage[feature] = {rank: dict[feature].get(rank)/sum*100}
                else:    
                    component_percentage[feature][rank] = dict[feature].get(rank)/sum*100

        return component_percentage


obj = FaithfulnessScore()
print(f'the faithful score is: {obj._compute_faithfulness_auc()}')
#print(f'top k features: {obj.top_k_features}')
#print(f'top_K_features_for_a_random_row: {obj._top_k_features_of_row()}')
#print(f'component percentages: {obj._component_percentage(obj.top_k_features_row)}')
