
import pandas as pd #e.g. pandas, sklearn, .....
import warnings # DO NOT modify this line
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning # DO NOT modify this line
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
warnings.filterwarnings("ignore", category=ConvergenceWarning) # DO NOT modify this line


class BankLogistic:
    def __init__(self, data_path): # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=',')
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Q1(self): # DO NOT modify this line
        """
        Problem 1:
            Load ‘bank-st.csv’ data from the “Attachment”
            How many rows of data are there in total?

        """
        # TODO: Paste your code here
        return self.df.shape[0]


    def Q2(self): # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        # TODO: Paste your code here
        num = self.df.select_dtypes(include='number').shape[1]
        cat = self.df.select_dtypes(include='object').shape[1]
        return (num,cat)
    
    def Q3(self): # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        # TODO: Paste your code here
        noyes =self.df.y.value_counts(normalize=True, dropna=True)
        return (round(noyes.loc['no'], 3), round(noyes.loc['yes'], 3))
      
    

    def Q4(self): # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        # TODO: Paste your code here
        df = self.df
        df.drop_duplicates(inplace=True)
        return df.shape
        

    def Q5(self): # DO NOT modify this line
        """
        Problem 5:
            5. Replace unknown value with null
            6. Remove features with more than 99% flat values. 
                Hint: There is only one feature should be drop
            7. Split Data
            -	Split the dataset into training and testing sets with a 70:30 ratio.
            -	random_state=0
            -	stratify option
            return the tuple of shapes of X_train and X_test.

        """
        # TODO: Paste your code here
        df = self.df
        #Q4
        df.drop_duplicates(inplace=True)
        #Q5
        df.replace('unknown',  np.nan, inplace=True)
        th = 0.99
        flat_col = [col for col in df.columns if df[col].value_counts(normalize=True).iloc[0] > th]
        df.drop(columns=flat_col,inplace=True)

        y = df.y
        X = df.drop(columns='y')
        X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3, random_state=0)
        return (X_train.shape,X_test.shape)

       
    def Q6(self): 
        """
        Problem 6: 
            8. Impute missing
                -	For numeric variables: Impute missing values using the mean.
                -	For categorical variables: Impute missing values using the mode.
                Hint: Use statistics calculated from the training dataset to avoid data leakage.
            9. Categorical Encoder:
                Map the ordinal data for the education variable using the following order:
                education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7} 
                Hint: Use One hot encoder or pd.dummy to encode ordinal category
            return the shape of X_train.

        """
        # TODO: Paste your code here
        education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7} 
        df = self.df
        #Q4
        df.drop_duplicates(inplace=True)
        #Q5
        df.replace('unknown',  np.nan, inplace=True)
        th = 0.99
        flat_col = [col for col in df.columns if df[col].value_counts(normalize=True).iloc[0] > th]
        df.drop(columns=flat_col,inplace=True)
        
        y = df.y
        X = df.drop(columns='y')
        X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3, random_state=0)
        #Q6
        num_imp = SimpleImputer(missing_values=np.nan,strategy='mean')
        cat_imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
        
        num = X_train.select_dtypes(include='number')
        cat = X_train.select_dtypes(include='object')
        X_train[num.columns] = pd.DataFrame(num_imp.fit_transform(num),columns=num.columns, index=X_train.index)
        X_train[cat.columns] = pd.DataFrame(cat_imp.fit_transform(cat),columns=cat.columns, index=X_train.index)
        X_train['education'] = X_train.education.map(education_order)
        dummy = pd.get_dummies(X_train[cat.columns])
        X_train.drop(columns = cat.columns,inplace=True)
        X_train = pd.concat([X_train,dummy],axis=1)
        return X_train.shape
    
    def Q7(self):
        ''' Problem7: Use Logistic Regression as the model with 
            random_state=2025, 
            class_weight='balanced' and 
            max_iter=500. 
            Train the model using all the remaining available variables. 
            What is the macro F1 score of the model on the test data? in 2 digits
        '''
        # TODO: Paste your code here
        education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7} 
        df = self.df
        #Q4
        df.drop_duplicates(inplace=True)
        #Q5
        df.replace('unknown',  np.nan, inplace=True)
        th = 0.99
        flat_col = [col for col in df.columns if df[col].value_counts(normalize=True).iloc[0] > th]
        df.drop(columns=flat_col,inplace=True)
        
        y = df.y
        X = df.drop(columns='y')
        X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3, random_state=0)
        #Q6
        num_imp = SimpleImputer(missing_values=np.nan,strategy='mean')
        cat_imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
        
        num = X_train.select_dtypes(include='number')
        cat = X_train.select_dtypes(include='object')
        X_train[num.columns] = pd.DataFrame(num_imp.fit_transform(num),columns=num.columns, index=X_train.index)
        X_train[cat.columns] = pd.DataFrame(cat_imp.fit_transform(cat),columns=cat.columns, index=X_train.index)
        X_train['education'] = X_train.education.map(education_order)
        dummy = pd.get_dummies(X_train[cat.columns])
        X_train.drop(columns = cat.columns,inplace=True)
        X_train = pd.concat([X_train,dummy],axis=1)
        
        #Q7
        num = X_test.select_dtypes(include='number')
        cat = X_test.select_dtypes(include='object')
        X_test[num.columns] = pd.DataFrame(num_imp.transform(num),columns=num.columns, index=X_test.index)
        X_test[cat.columns] = pd.DataFrame(cat_imp.transform(cat),columns=cat.columns, index=X_test.index)
        X_test['education'] = X_test.education.map(education_order)
        dummy = pd.get_dummies(X_test[cat.columns])
        X_test.drop(columns = cat.columns,inplace=True)
        X_test = pd.concat([X_test,dummy],axis=1)
        lr = LogisticRegression(random_state=2025,class_weight='balanced',max_iter=500)
        lr.fit(X_train,y_train)
        pred = lr.predict(X_test)
        return round(classification_report(y_test,pred,output_dict=True)['macro avg']['f1-score'],2)
        

test = BankLogistic('bank-st.csv')
print(test.Q7())
   