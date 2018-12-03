#!/usr/bin/python
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import math

def create_df(filename):
    # Read the train set
    df = pd.read_csv(filename)

    # Combine SigSp (# siblings on board and Parch (# parents/children on board) in 1 column
    df['Family'] = df['SibSp'] + df['Parch']

    # Encode sex column to a number
    sex_encoder = preprocessing.LabelEncoder()
    sex_encoder.fit(df['Sex'])
    sex_encoded = sex_encoder.transform(df['Sex'])
    df['SexEncoded'] = sex_encoded

    # Find average age used to replace NaN ages
    avg_age = df['Age'].mean()

    # Replace NaN age with average age
    df['Age'].fillna(avg_age, inplace=True)

    # TODO: Keep cabin letter to note location of passenger
    # Check if passenger has a cabin
    df['HasCabin'] = [0 for x in range(len(df))]
    for i in range(len(df)):
        if pd.isna(df['Cabin'][i]):
            df.at[i, 'HasCabin'] = 0
        else:
            df.at[i, 'HasCabin'] = 1
    return df


def run(train_data):
    # Some features are not important (like passenger ID and name)
    features = ['Pclass', 'SexEncoded', 'Age', 'Family', 'HasCabin']
    target = train_data['Survived']

    clf = RandomForestClassifier(n_estimators=500, max_depth=4)
    #clf = SVC(kernel="linear")

    # Perform 10-fold cross validation to evaluate the classifier
    scores = cross_val_score(clf, train_data[features], target, cv=10, scoring="accuracy")
    return list(scores)


def predict(train_data, test_data):
    features = ['Pclass', 'SexEncoded', 'Age', 'Family', 'HasCabin']
    target = train_data['Survived']

    clf = RandomForestClassifier(n_estimators=500, max_depth=4)
    clf.fit(train_data[features], target)
    predictions = clf.predict(test_data[features])

    print "[+] Writing results to predictions.csv"
    pd.DataFrame({"PassengerId": test_data['PassengerId'], "Survived": predictions}).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    train_data = create_df("train.csv")

    if raw_input("Perform cross validation of train.csv? (y/n)\n> ") == "y":
        scores = run(train_data)
        print "[*] 10-fold cross validation results:"
        for val in scores:
            print "\t" + str(val)
        print "[*] Average: " + str(sum(scores) / float(len(scores)))

    if raw_input("Perform prediction of test.csv? (y/n)\n> ") == "y":
        test_data = create_df("test.csv")
        predict(train_data, test_data)
