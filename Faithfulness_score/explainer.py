### LIME explanations ###
import pandas as pd
import time
import lime
import lime.lime_tabular
import numpy as np
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Explainer():
    def __init__(self, mode):
        self.not_trained = False #run with not_trained = True one time, set trained = False after. This is for the saving the black-box model

        self.data = r"..\dataset\demographics_to_placement_simulation_WR_no_cap_continuous_2.csv"

        self.filter_employment = True
        self.drop_minor = True
        self.drop_imbalanced = True
        self.cost_multiplication = True

        self.hyparameter = {'max_depth' : 7, 'max_features' : 'sqrt', 'min_samples_leaf' : 5, 
        'min_samples_split' : 5, 'n_estimators' : 100}
        self.model, self.test_data, self.labels, self.X_background = self.engineer_features()
        self.class_names = np.array([0,1,2])

        if mode == 'LIME':
            self.explainer_model = lime.lime_tabular.LimeTabularExplainer(self.test_data, feature_names=list(self.labels), class_names=self.class_names, discretize_continuous=False)
        elif mode =='SHAP':
            #shap.KernelExplainer
            #self.explainer_model = shap.KernelExplainer(self.model.predict_proba, data = self.X_background )
            #shap.TreeExplainer
            self.explainer_model = shap.TreeExplainer(self.model)
        self.cleaned_data

        self.feature_names_dict = {0: 'age', 1: 'num_previous_recidivisms', 2: 'race_Caucasian', 3: 'race_Hispanic',
       4: 'gender_Male', 5: 'HighestEducationLevel_High_School_Diploma',
       6: 'HighestEducationLevel_No_HS_Diploma',
       7: 'HighestEducationLevel_Some_College', 8: 'c_vio', 9: 'Employed',
       10: 'weighted_jl_total', 11: 'weighted_wr_total', 12: 'weighted_hd_total'}

        #the below are for LIME
        #print(self.explainer.explain_instance(np.array(self.test_data[self.test_data.index == 9])[0], self.model.predict_proba, num_features=14, top_labels=3))
        #print(self.explainer.feature_names)

    def engineer_features(self):
        continuous_after = pd.read_csv(self.data)
        if self.filter_employment:
            unknown_employment_index = continuous_after[ (continuous_after['employmentStatus_Full_Time'] == 0) & (continuous_after['employmentStatus_Part_Time'] == 0) & (continuous_after['employmentStatus_Unemployed'] == 0)].index
            unknown_employment_age = continuous_after[ (continuous_after['employmentStatus_Part_Time'] == 0) & (continuous_after['employmentStatus_Unemployed'] == 0) & (continuous_after['employmentStatus_Full_Time'] == 0) ]['age']
            unemployment_index = continuous_after[ continuous_after['employmentStatus_Unemployed'] == 1].index
            unknown_employment_index

            type1_index = list(unemployment_index) + list(unknown_employment_index)
            type2_index = list(continuous_after[ continuous_after['employmentStatus_Full_Time'] == 1].index) + list(continuous_after[ continuous_after['employmentStatus_Part_Time'] == 1].index)
            new_row = [0] * continuous_after.shape[0]
            for index in type1_index:
                new_row[index] = 1

            continuous_after['Employed'] = new_row
            continuous_after.drop(['employmentStatus_Part_Time', 'employmentStatus_Full_Time'], axis =1, inplace = True )

            new_row = [0] * continuous_after.shape[0]
            for index in type2_index:
                new_row[index] = 1

            continuous_after['Unemployed'] = new_row
            continuous_after.drop(['employmentStatus_Unemployed'], axis =1, inplace = True )
            if self.drop_minor:
                continuous_after.drop(['Unemployed','ocp_wr_low_los', 'ocp_jl_low_los', 'ocp_hd_low_los','licenseStatus_Not_Suspended','licenseStatus_Suspended'], axis=1, inplace=True) 

        # one hot encoding  -> the target is "placement", so we don't need to encode this column, also bcz RandomForest can handle categorical target data
        # work release = 0 | home detention = 1 | jail = 2
        for ind, item in continuous_after.iterrows():
            if item['placement'] == 'work release':
                continuous_after['placement'].iat[ind] = '0'
            elif item['placement'] == 'home detention':
                continuous_after['placement'].iat[ind] = '1'
            elif item['placement'] == 'jail':
                continuous_after['placement'].iat[ind] = '2'

        #object to int
        continuous_after['placement'] = pd.to_numeric(continuous_after['placement'])

        # drop imbalanced data
        if self.drop_imbalanced:
            continuous_after.drop(['race_Other', 'registeredSexOffender_TRUE', 'violentOffender_TRUE', 'gangMember_x_TRUE', 'homeless_TRUE'], axis = 1, inplace = True)
           
        #cost multiplication
        if self.cost_multiplication:
            continuous_after['weighted_jl_total'] = continuous_after['c_occu_jail']*continuous_after['ocp_jl_total']
            continuous_after['weighted_wr_total'] = continuous_after['c_occu_wr']*continuous_after['ocp_wr_total']
            continuous_after['weighted_hd_total'] = continuous_after['c_occu_hd']*continuous_after['ocp_hd_total']
            continuous_after.drop(['c_occu_jail', 'c_occu_wr', 'c_occu_hd', 'ocp_jl_total', 'ocp_wr_total', 'ocp_hd_total'], axis = 1, inplace=True)
        self.cleaned_data = continuous_after

        y2 = continuous_after["placement"]
        X2 = continuous_after.drop(["placement"], axis = 1)

        X_train, X_test, y_train, y_test = train_test_split( X2, y2, train_size = 0.7, random_state = 40) #random_state must be 40 for alignment between files 
        

        # if already trained, load the model
        if (self.not_trained): 
            rf_model2 = RandomForestClassifier(max_depth= self.hyparameter['max_depth'], max_features = self.hyparameter['max_features'],
                                            min_samples_leaf = self.hyparameter['min_samples_leaf'], min_samples_split = self.hyparameter['min_samples_split'],
                                            n_estimators = self.hyparameter['n_estimators'], random_state= 50) 
            rf_model2.fit(X_train, y_train)
            with open('rf_model2.pkl', 'wb') as file:
                pickle.dump(rf_model2, file)
        else:
            with open('rf_model2.pkl', 'rb') as file:
                rf_model2 = pickle.load(file)
        
        return rf_model2, X_test, continuous_after.columns.drop(['placement']), self.generate_X_background_Kernal(X_train)

    @staticmethod
    def generate_X_background_Kernal(X_train):
        X_background = X_train.iloc[np.random.choice(X_train.shape[0], 50, replace=False)]
        return X_background



    