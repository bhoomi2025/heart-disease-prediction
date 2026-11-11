# Heart Disease Prediction

Machine Learning project jo patient ki details se heart disease ka risk predict karta hai.

## Dataset
Heart Failure Prediction dataset (`heart.csv`), jisme 11 features aur target column `HeartDisease` hai.

## Steps
1. EDA (histograms, countplots, boxplot, heatmap)
2. Cleaning: Cholesterol aur RestingBP ke 0 values ko mean se replace kiya
3. Encoding: `pd.get_dummies`
4. Scaling: `StandardScaler`
5. Models: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM
6. KNN model ko `joblib` se save karke Streamlit app banaya



## Author
bhoomi