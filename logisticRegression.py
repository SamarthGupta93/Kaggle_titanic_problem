# Import libraries
import pandas as pd
import numpy as np

def load_data(file_path):
    return = pd.read_csv(file_path)  
  
# Load training data  
train_data = load_data('train.csv')
train_data.head()

# Load test data
test_data = load_data('test.csv')
test_data.head()

# Load the labels for test data
test_data_survived = pd.read_csv('gender_submission.csv')

# Check for columns with null values
train_data.columns[train_data.isnull().any()]

train_data.info()
test_data.info()

# Analyse the data set and select columns to keep
num_columns_in_rawdata = len(train_data.columns)
columns_to_keep = [2] + list(range(4,8)) + list(range(9,10)) + [11]
columns_to_keep_test = [1] + list(range(3,7)) + [8] + [10]

# Set X_train, X_test, y_train, y_test
X_train = train_data.ix[:,columns_to_keep]
y_train = train_data['Survived']
X_test = test_data.ix[:,columns_to_keep_test]
y_test = test_data_survived['Survived']

# Data Preprocessing
X_train['Sex'][X_train['Sex']=='male'] = 1
X_train['Sex'][X_train['Sex']=='female'] = 2
X_test['Sex'][X_test['Sex']=='male'] = 1
X_test['Sex'][X_test['Sex']=='female'] = 2

X_train.Embarked.unique()
X_train['Embarked'].value_counts()
X_train['Embarked'][X_train['Embarked'].isnull()] = 'S'

X_train.isnull().sum()

X_train['Age'][X_train['Sex']==1].mode()
X_train['Age'][X_train['Age'].isnull()] = 25
X_train.isnull().sum()

#convert Embarked values to integers
X_train['Embarked'][X_train['Embarked']=='S'] = 1
X_train['Embarked'][X_train['Embarked']=='C'] = 2
X_train['Embarked'][X_train['Embarked']=='Q'] = 3

X_test['Embarked'][X_test['Embarked']=='S'] = 1
X_test['Embarked'][X_test['Embarked']=='C'] = 2
X_test['Embarked'][X_test['Embarked']=='Q'] = 3

#process test data: remove null values from age and fare
#Age
X_test['Age'].mean()
#Setting age as average value
X_test['Age'][X_test['Age'].isnull()] = 30

#Set null values for fare
#Set the fare value as average of the fares of that Pclass
X_test[X_test['Fare'].isnull()]

#Determine average fare for Pclass=3 (only Pclass = 3 has a null fare)
X_test['Fare'][X_test['Pclass']==3].mean()

#Set null value of fair as average value
X_test['Fare'][X_test['Fare'].isnull()] = 12.45

X_test.info()
X_train.info()

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Run Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=100).fit(X_train,y_train)

# Accuracy of Logistic Regression
print('Accuracy of Regression on training set: ', clf.score(X_train,y_train))
print('Accuracy of Regression on test set: ', clf.score(X_test,y_test))
