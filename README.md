# Kaggle_titanic_problem
Predicts if a person on the ship is likely to survive

The labels are provided in gender_submission.csv
The train and test csv files are also provided

Data preprocessing
> Removed null values from Age, Embarked and Fare
> For embarked - replacing null with 'S', since it has significantly higher frequency than the other two categories
> Age (Null values) is taken care of separately for Men and Women and it the average of the modes in both categories
> For Fare, taking the average fare of each class to fill up the null values

Feature scaling

Define and Run logistic regression with C=100

Calculate the accuracy on test and train data

Check for overfitting/underfitting
