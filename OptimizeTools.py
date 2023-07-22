import pandas as pd
import numpy as np


def to_csv_sample_sub_df(n_predictions):
    # This function creates a sample submission dataframe and shows
    # its first records via predicted results from RF model
    sample_submission_df = pd.read_csv("TitanicSpaceshipDatasets/sample_submission.csv")
    sample_submission_df['Transported'] = n_predictions
    sample_submission_df.to_csv('working/submission.csv', index=False)
    print(sample_submission_df.head())


def NaN_to_zero(df, cols):
    # This function takes a DataFrame & its target volumes for
    # converting its NaN values to zero.
    df[cols] = df[cols].fillna(value=0)


def split_and_drop_col(df, cols, col, by):
    # Divides a column into smaller components.
    df[cols] = df[col].str.split(by, expand=True)
    df = df.drop(col, axis=1)   # Drop the target column.


def bool_to_binary(df, cols):
    for col in cols:
        df[col] = df[col].astype(int)