"""
Titanic Spaceship Challenge
Author: Mohammad Matin Kateb
https://www.kaggle.com/competitions/spaceship-titanic
"""

import pandas as pd
import tensorflow_decision_forests as tfdf
import OptimizeTools as ot


def main():
    # Load the test & train datasets
    test_df  = pd.read_csv('TitanicSpaceshipDatasets/test.csv')
    train_df = pd.read_csv('TitanicSpaceshipDatasets/train.csv')

    # Replace NaN values with zero
    ot.NaN_to_zero(test_df,  ['VIP', 'CryoSleep'])
    ot.NaN_to_zero(train_df, ['VIP', 'CryoSleep'])

    # Creating New Features - Deck, Cabin_num and Side from the column Cabin and remove Cabin
    ot.split_and_drop_col(test_df,  ['Deck', 'Cabin_num', 'Side'], 'Cabin', '/')
    ot.split_and_drop_col(train_df, ['Deck', 'Cabin_num', 'Side'], 'Cabin', '/')

    # Convert boolean to 1's and 0's
    ot.bool_to_binary(test_df,  ['VIP', 'CryoSleep'])
    ot.bool_to_binary(train_df, ['VIP', 'CryoSleep', 'Transported'])

    # Convert pd dataframe to tf dataset
    test_ds  = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="Transported")

    rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")
    rf.compile(metrics=['accuracy'])
    rf.fit(x=train_ds)

    # Get the predictions for testdata
    predictions = rf.predict(test_ds)
    n_predictions = (predictions > 0.5).astype(bool)

    ot.to_csv_sample_sub_df(n_predictions)



if __name__ == "__main__":
    main()