How to use this model framework:

cd to this folder: Generative-AI-and-LLM-in-Healthcare-Operations\Model_framework>
run this cmd python -m main_script --dataset <either trajectory or placement> --model <one of 5 models> --actions <1-6>
<one of 5 models>: rf, logreg, mlp, adaboost, svm
<actions 1-6>: 
1. Save model
2. Prediction (fit, predict, score, save)
3. Grid Search
4. SHAP values
5. LIME values
6. All

For example: if I want to find the best combination of parameters (grid_search) of svm for the risk prediction dataset, I run:
python -m main_script --dataset trajectory --model svm --actions 3