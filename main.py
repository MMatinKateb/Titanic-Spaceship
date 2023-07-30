"""
Titanic Spaceship Challenge
Author: Mohammad Matin Kateb
https://www.kaggle.com/competitions/spaceship-titanic
"""

import pandas as pd
import OptimizeTools as ot
import ExamineData as ed
from os import system
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

def main():
    try:
        system('clear')
    except:
        system('CLS')

    print("[+] Performing the model...")

    # Load the test & train datasets
    test_df = pd.read_csv('TitanicSpaceshipDatasets/test.csv')
    train_df = pd.read_csv('TitanicSpaceshipDatasets/train.csv')

    # Get rid of string columns
    train_df = ed.exclude_string_columns(train_df)
    test_df = ed.exclude_string_columns(test_df)

    # Feature Selection
    x_train = train_df.drop('Transported', axis=1)
    y_train = train_df['Transported']
    x_test = test_df

    ot.NaN_to_zero(x_train, x_train.columns)
    ot.NaN_to_zero(x_test, x_test.columns)

    # Model Training and Prediction with hyperparameters
    model1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model2 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
    model3 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

    # Create a Voting Classifier with multiple models
    voting_model = VotingClassifier(estimators=[('gb', model1), ('lr', model2), ('rf', model3)], voting='soft')

    # Fit the model to the training data and make predictions on the test data
    voting_model.fit(x_train, y_train)
    y_pred = voting_model.predict(x_test)

    # Create submission file
    ot.to_csv_sample_sub_df(y_pred)



if __name__ == "__main__":
    main()