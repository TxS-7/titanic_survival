#!/usr/bin/python
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import math

def run():
    # Read the train set
    train_data = pd.read_csv("train.csv")

    # Combine SigSp (# siblings on board and Parch (# parents/children on board) in 1 column
    train_data['Family'] = train_data['SibSp'] + train_data['Parch']

    # Encode sex column to a number
    sex_encoder = preprocessing.LabelEncoder()
    sex_encoder.fit(train_data['Sex'])
    sex_encoded = sex_encoder.transform(train_data['Sex'])
    train_data['SexEncoded'] = sex_encoded

    # Find average age used to replace NaN ages
    avg_age = train_data['Age'].mean()

    # Replace NaN age with average age
    train_data['Age'].fillna(avg_age, inplace=True)

    # TODO: Keep cabin letter to note location of passenger
    # Check if passenger has a cabin
    train_data['HasCabin'] = [0 for x in range(len(train_data))]
    for i in range(len(train_data)):
        if pd.isna(train_data['Cabin'][i]):
            train_data.at[i, 'HasCabin'] = 0
        else:
            train_data.at[i, 'HasCabin'] = 1


    # Some features are not important (like passenger ID and name)
    features = ['Pclass', 'SexEncoded', 'Age', 'Family', 'HasCabin']
    target = train_data['Survived']

    clf = RandomForestClassifier(n_estimators=500, max_depth=4)
    #clf = SVC(kernel="linear")

    # Perform 10-fold cross validation to evaluate the classifier
    scores = cross_val_score(clf, train_data[features], target, cv=10, scoring="accuracy")
    return list(scores)

if __name__ == "__main__":
    scores = run()

    print "[*] 10-fold cross validation results:"
    for val in scores:
        print "\t" + str(val)
    print "[*] Average: " + str(sum(scores) / float(len(scores)))
