import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic_to_student.csv) and answer the questions.
    (Note that the following functions already take the Titanic dataset as a DataFrame, so you don’t need to use read_csv.)

"""


def Q1(df):
    """
        Problem 1:
            How many rows are there in the "titanic_to_student.csv"?
    """
    # TODO: Code here
    
    return df.shape[0]


def Q2(df):
    '''
        Problem 2:
            Drop unqualified variables
            Drop variables with missing > 50%
            Drop categorical variables with flat values > 70% (variables with the same value in the same column)
            How many columns do we have left?
    '''
    # TODO: Code here
    df1 = df.dropna(axis=1, thresh=df.shape[0]/2)
    cat_col = df1.select_dtypes(include = 'object')
    drop_col = [col for col in cat_col.columns if cat_col[col].value_counts(normalize = True).max()>0.7]
    df = df1.drop(axis=1, columns = drop_col)
    return df.shape[1]


def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    # TODO: Code here
    df1 = df.dropna(subset=['Survived'])
    return df1.shape[0]


def Q4(df):
    '''
       Problem 4:
            Handle outliers
            For the variable “Fare”, replace outlier values with the boundary values
            If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
            If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
            What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    # TODO: Code here
    q75, q25 = np.percentile(df.Fare.dropna(), [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    df.loc[df["Fare"] < min, 'Fare'] = min
    df.loc[df["Fare"] > max, 'Fare'] = max
    return round(df.Fare.mean(),2)


def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    # TODO: Code here
    num_imp=SimpleImputer(missing_values=np.nan, strategy='mean')
    df['Age']= pd.DataFrame(num_imp.fit_transform(df[['Age']]))
    return round(df.Age.mean(),2)


def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    # TODO: Code here
    dummy_df = pd.get_dummies(df['Embarked'], drop_first=False)
    return round(dummy_df.Q.mean(),2)


def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection, 
            Don't forget to impute missing values with mean.
    '''
    # TODO: Code here
    num_imp=SimpleImputer(missing_values=np.nan, strategy='mean')
    df['Survived']= pd.DataFrame(num_imp.fit_transform(df[['Survived']]))
    df['Survived']= df['Survived'].apply(lambda x: 1.0 if x>0.5 else 0.0)
    y = df.pop('Survived')
    X = df
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3, random_state=123)
    p_survived = y_train.sum() / y_train.shape[0]
    return round(p_survived,2)
df = pd.read_csv("titanic_to_student.csv")
print(Q7(df))