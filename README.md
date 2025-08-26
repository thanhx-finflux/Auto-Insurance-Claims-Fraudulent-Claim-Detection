# Auto-Insurance-Claims-Fraudulent-Claim-Detection
This project implementations a fraud detection system for auto insurance claims data using Logistic Regression and Random Forest.
This report details the steps taken to build and evaluate two machine learning models to detect fraudulent insurance claims for Global Insure Company. Using historical claim data of 1000 claims with 40 features, we aimed to classify claims as either fraudulent or legitimate. The process included data cleaning, exploratory data analysis, feature engineering, model building, feature selection, optimization, and validation.
Model evaluation with a 70-30 train-validation split was conducted to assess the performance of the models. 

Model 1: Logistic Regression
Model 2: Random Forest Classifier

Key metrics such as accuracy, precision, recall, and F1-score were calculated to determine the effectiveness of each model in identifying fraudulent claims. The results showed that the Random Forest Classifier outperformed the Logistic Regression model in key metrics. Key fraudulent patterns were also identified through feature importance analysis, like the high importance of features such as hobbies and claim severity, which can inform future prevention and early detection to reduce financial losses.

1. Logistic Regression

- Feature selection: Identify and select 46 important features using RFECV

- Utilize these features to build a Logistic Regression model, evaluate its performance to detect multicollinearity and ensure that the selected features are not high correlation and VIF. 12 features with VIF < 5 and p-value < 0.05 were retained in the model. 

- insured_hobbies_board-games, insured_occupation_exec-managerial, insured_relationship_not-in-family, insured_relationship_other-relative... are high positive correlation in the model.

- Training performance:
    + Accuracy: 0.6905
    + Sensitivity: 0.6343
    + Specificity: 0.7467
    + Precision: 0.7146
    + Recall: 0.6343
    + F1 Score: 0.6720
    + AUC-ROC: 0.7585

The model is moderate performance with balanced sensitivity (0.6343) and specificity (0.7467), but high false negatives (192) indicate missed fraudulent claims.

- Optimal cutoff: 0.52 based on maximum accuracy, maintaining sensitivity and specificity.
    + Accuracy: 0.6952 
    + Sensitivity: 0.6343
    + Specificity: 0.7562
    + Precision: 0.7223
    + Recall: 0.6343
    + F1 Score: 0.6755
    + AUC-ROC: 0.7585

The cutoff point for classifying claims as fraudulent is set at 0.52.

- Validation performance based on cutoff 0.52:
    + Accuracy: 0.6400
    + Sensitivity: 0.3919
    + Specificity: 0.7212
    + Precision: 0.3152
    + Recall: 0.3919
    + F1-Score: 0.3494
    + ROC-AUC: 0.5928

The model shows a drop in performance on validation data, especially in sensitivity (0.3919) and precision (0.3152). This indicates that the model may not generalize well to unseen data, leading to a higher rate of false negatives. Significant overfitting (training performance much better than validation), f1-score in training (0.6755) is much higher than f1-score in validation (0.3494).

2. Random Forest Model
- Select 0.90 important features using feature_importances_ from the base random forest model. 30 features were selected to build the model.

- Training performance (base):
    + Accuracy: 1.0000
    + Sensitivity: 1.0000
    + Specificity: 1.0000
    + Precision: 1.0000
    + Recall: 1.0000
    + F1-Score: 1.0000
    + ROC-AUC: 1.0000

The model demonstrates perfect performance on the training data, indicating potential overfitting. The model may not generalize well to unseen data, leading to a higher false negatives.

- Average cross validated scoring = f1-weighted 0.9237, with standard deviation 0.0109, indicating model robustness, with minimal variance across folds.

- Hyperparameter tuning using GridSearchCV identified optimal parameters:  {'bootstrap': False, 'max_depth': 8, 'max_features': 0.4, 'min_samples_leaf': 20, 'min_samples_split': 25, 'n_estimators': 100}.These parameters were selected to improve model performance and reduce overfitting. The model was retrained using these optimal parameters and achieved the following performance metrics:

    + Accuracy: 0.8876
    + Sensitivity: 0.9067
    + Specificity: 0.8686
    + Precision: 0.8734
    + Recall: 0.9067
    + F1-Score: 0.8897
    + ROC-AUC: 0.9700

The tuned model shows significant improvement, with reduced overfitting and better generalization to unseen data. The performance metrics indicate a well-balanced model, particularly in terms of sensitivity and specificity.

- Validation performance based on tuned parameters:
    + Accuracy: 0.8167
    + Sensitivity: 0.7568
    + Specificity: 0.7212
    + Precision: 0.6022
    + Recall: 0.7568
    + F1-Score: 0.6707
    + ROC-AUC: 0.7559

The model maintains good performance on the validation data, with a balanced sensitivity (0.7568) and specificity (0.7212). The precision (0.6022) indicates that while the model is effective at identifying fraudulent claims, there is tradeoff between precision and recall. The ROC-AUC score of 0.7559 suggests that the model has a good ability to distinguish between fraudulent and legitimate claims. This model is preferred over logistic regression due to its better performance and generalization with unseen data, particularly in terms of sensitivity and specificity, with reduced overfitting and improved robustness, with a more balanced approach to identifying fraudulent claims.

- Model comparison random forest outperforms logistic regression in all most key metrics:
    + Accuracy: 0.8167 vs 0.6400
    + Sensitivity: 0.7568 vs 0.3919
    + Specificity: 0.7212 vs 0.7212
    + Precision: 0.6022 vs 0.3152
    + Recall: 0.7568 vs 0.3919
    + F1-Score: 0.6707 vs 0.3494
    + ROC-AUC: 0.7559 vs 0.5928

The random forest model is higher performing in all key metrics except for specificity, where both models are equal. In terms of overall effectiveness at identifying fraudulent claims, the random forest model demonstrates better performance and generalization capabilities compared to logistic regression.

The logistic regression model, while simple, may not capture complex patterns, and non-linear relationships as effectively as the random forest model.
It is very poor validation performance and high bias, leading to underfitting, particularly in sensitivity and recall.

Random forest model, while more complex, is better suited for this classification task due to its ability to handle non-linear relationships and interactions between features. It is higher performing in all key metrics, particularly in sensitivity or recall, which are crucial for fraud detection.


## Author
Thanh Xuyen Nguyen https://www.linkedin.com/in/xuyen-thanh-nguyen-0518/
