# Predicting Problematic Internet Use in Children Using Physical Activity Data

## University of Arizona - INFO531 Data Warehousing and Analytics in the Cloud


#### Problem
Excessive internet and technology use among children can lead to problematic behaviors, but early signs are often difficult to detect. This project aims to predict levels of problematic internet use by analyzing children’s physical activity data, enabling early identification and intervention to promote healthier digital habits and responsible technology use.


#### Data source
The project uses data from the Healthy Brain Network (HBN) dataset, which includes health, physical activity, and internet usage information for 5,000 individuals aged 5–22. The target variable is the Severity Impairment Index (SII), measuring problematic internet use. The dataset, available on Kaggle, includes train.csv and test.csv files.


#### Approach
The project follows a structured data preparation and analysis workflow. First, data discovery involves understanding the dataset, checking variable types, distributions, correlations, and identifying quality issues such as missing values, duplicates, or inconsistencies. Next, data cleaning and transformation addresses these issues by handling missing values (imputation or removal), standardizing and normalizing variables, converting categorical data to numerical, correcting outliers, and removing or creating relevant features. Finally, the processed data is used for exploratory data analysis, statistical evaluation, and machine learning modeling to predict problematic internet use in children.


#### Key results
The data preparation phase resulted in cleaned and processed training (3960 records, 82 features) and testing datasets (20 records, 59 features) with no missing values. The target variable, Severity Impairment Index (SII), was categorized into four levels: None, Mild, Moderate, and Severe. Correlation analysis highlighted relationships between predictors and the response, guiding feature selection.

Several machine learning models were trained and evaluated to predict problematic internet use based on physical activity and related features:
1. Decision Tree – produced categorical predictions with associated performance metrics and confusion matrix.
2. Random Forest – ensemble model improved classification accuracy over a single decision tree.
3. Gradient Boosting – iterative ensemble approach enhanced prediction performance.
4. K-Nearest Neighbors (KNN) – classified observations based on nearest feature neighbors.

All models were validated using train-test splits, with performance assessed via confusion matrices and standard classification metrics, demonstrating the feasibility of predicting levels of problematic internet use from the dataset.


#### Conclusion
Among the models evaluated, Random Forest provided the best balance of high performance and low overfitting risk, achieving an accuracy of 99.6% along with strong precision, recall, and F-scores. While Decision Tree and Gradient Boosting achieved perfect scores, they risk overfitting. K-Nearest Neighbors performed the worst, with significantly lower accuracy and metric scores. Overall, Random Forest is the recommended model for predicting levels of problematic internet use in children based on physical activity data.


#### Technical stack
The project was implemented in **Python** using **Jupyter Notebook (Anaconda)**. Key libraries and tools included:
* **Data manipulation & analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Statistical analysis:** `scipy`, `sklearn`
* **Machine learning models:** `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`
* **Model evaluation & validation:** `train_test_split`, `KFold`, `cross_val_score`, `confusion_matrix`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score`, `roc_auc_score`, `classification_report`


#### Data visualisations


<img width="2464" height="2411" alt="image" src="https://github.com/user-attachments/assets/b0f995e6-fbb6-45ff-88ad-4599637fc2c7" />


<img width="649" height="545" alt="image" src="https://github.com/user-attachments/assets/9f27ddbb-a016-48bc-b8eb-53e919b585ed" />


<img width="649" height="545" alt="image" src="https://github.com/user-attachments/assets/1df7e415-ecf2-46be-8f2d-7d61720d111c" />


<img width="649" height="545" alt="image" src="https://github.com/user-attachments/assets/79be93da-1311-4a05-a19f-55e8d301338f" />


<img width="649" height="545" alt="image" src="https://github.com/user-attachments/assets/03ed8dc0-ebaa-4e3c-bbd0-38cfaa646a83" />












