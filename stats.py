#!/usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt

def create_graph(df, feature):
    died_df = df[df['Survived'] == 0][feature].value_counts()
    survived_df = df[df['Survived'] == 1][feature].value_counts()
    combined = pd.DataFrame([died_df, survived_df])
    ax = combined.plot.bar(title="Survival based on: " + feature)
    plt.xticks([0, 1], ["Died", "Survived"], rotation="horizontal")
    return ax


def run():
    df = pd.read_csv("train.csv")

    # Show total number of people who survived / died
    df['Survived'].value_counts().plot.bar(title="Survival")
    plt.xticks([0, 1], ["Died", "Survived"], rotation="horizontal")

    # Gender of passenger
    create_graph(df, "Sex")

    # Passenger class
    create_graph(df, "Pclass")

    # Embarked location
    create_graph(df, "Embarked")

    # Family members on board
    df['Family'] = df['SibSp'] + df['Parch']
    create_graph(df, "Family")

    # Passenger has / doesn't have cabin
    df['HasCabin'] = [0 for x in range(len(df))]
    for i in range(len(df)):
        if pd.isna(df['Cabin'][i]):
            df.at[i, 'HasCabin'] = 0
        else:
            df.at[i, 'HasCabin'] = 1
    ax = create_graph(df, "HasCabin")
    ax.legend(["No cabin", "With cabin"])

    # Age groups survival graph
    # Find average age to replace NaN values
    avg_age = df['Age'].mean()
    df['Age'].fillna(avg_age, inplace=True)
    df['AgeGroups'] = [0 for x in range(len(df))]
    for i in range(len(df)):
        val = df['Age'][i]
        if val >= 18 and val <= 35:
            df.at[i, 'AgeGroups'] = 1
        elif val >= 36 and val <= 50:
            df.at[i, 'AgeGroups'] = 2
        elif val > 50:
            df.at[i, 'AgeGroups'] = 3
    ax = create_graph(df, "AgeGroups")
    ax.legend(["< 18", "18-35", "36-50", "> 50"])

    df.info()
    plt.show()

if __name__ == "__main__":
    run()
