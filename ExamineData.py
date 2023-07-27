import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import OptimizeTools as ot
from sklearn.linear_model import LogisticRegression

# This function updates the 'Transported' column with updated values
def modify_phase2_dataset():
    sub_df   = pd.read_csv('working/submission.csv')
    train_df = pd.read_csv('TitanicSpaceshipDatasets/train.csv')

    # Update 'Transported' column
    train_df = train_df.iloc[:4277]
    train_df['Transported'] = sub_df['Transported']
    train_df.to_csv('ExaminedData/Phase2-DS.csv', index=False)

    print("[***] Phase2-DS.csv successfully updated.")

def run_gradient_boosting(x_train, y_train, test_dataframe):

    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(test_dataframe)
    return predictions

def optimized_run_gradient_boosting(train_dataframe, test_dataframe):

    train_df       = pd.read_csv('TitanicSpaceshipDatasets/train.csv')
    test_df        = pd.read_csv('TitanicSpaceshipDatasets/test.csv')
    # Exclude string columns from test and phase-2 DataFrames
    test_df     = pd.concat( [test_df[test_df.select_dtypes(exclude=object).columns], test_df['VIP'],
                              test_df['CryoSleep']], axis=1 )
    train_df = pd.concat( [train_df[train_df.select_dtypes(exclude=object).columns], train_df['VIP'], 
                            train_df['CryoSleep']], axis=1 )

    ot.NaN_to_zero(train_df, train_df.columns)
    ot.NaN_to_zero(test_df, test_df.columns)

    x_train = train_df.drop('Transported', axis=1)
    x_train = train_df[x_train.columns]
    y_train = train_df['Transported']

    # Get rid of NaN values in test_df
    ot.NaN_to_zero(test_df, test_df.columns)

    y_pred = run_gradient_boosting(x_train, y_train, test_df)
    y_pred = (y_pred > 0.5).astype(bool)


def exclude_column(df, column_name):
    return df.drop(column_name, axis=1)

def exclude_string_columns(df):
    return df.select_dtypes(exclude=['object'])
