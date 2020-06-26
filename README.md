# Loan-Prediction

# Presentation :

This is a personnal project, the aim is to perform a binary classification on the Loan_Status feature of the Loan_prediction dataset (Can be download from this git or from Kaggle at the following url : https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)

This project is a part of a long term project which is to make this git a solid data science algorithm data base, implementation and explanation.
In this repo, you'll find some basic algorithm implementation and basic preprocessing techniques which you can re-use for your own purpose !


I've tested out 5 different algorithms so far : K_neighboor Classifier, LogisticRegression, DecisionTreeClassifier, RandomForest, SVM with Random giving the best result.

First part of the script is the testing of the different algorithm to find out which one provide the best results !

Second part is a gridsearching hyperparameters tuning for RandomForestClassifier, implementing pipelines and cross validation ! 

Each part can be executed separately !

Hope you enjoy!






# TODO :

- More data pre-processing (ie: Discuss missing value, for now num datas were changed by their median but i need to check if just removing the datas wouldn't provide better results  DONE: drop NA won't provide better results, still need to explore NA handling)
- More and more data pre-processing (ie : discussing encoding cat datas DONE : Tried a few thing, choose experimentally imputation + encoding.)
- More and more and more data pre-processing (ie: Feature selection) -- Correlation Matrix plotted, the dataset is not really good, we could do some up-scaling/down scaling process and feature engineering but i'll just do some basic pre-preprocessing process (imputation + scaling). I'll keep those stuff for some more interesting datasets!
- Hyper parameter tuning (DONE for RandomForest : Pipeline for process validation and hyperparameters tuning ! Still need to implement preprocessing gridsearch tho!)
- Add new models (RandomForest, SVC, ..) -- DONE SVC ADDED, RandomForest added
- Create pipelines for final process validation (DONE)
- Add prediction probabilities (DONE)
- Have fun (DONE : A thousand times ! :D)


