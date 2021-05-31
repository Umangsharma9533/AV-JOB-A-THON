# Importing all the libraries used
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot

from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pickle
sns.set()

# Function to perform Under Sampling and Over Sampling
# By default both will be performed and output is generated for both under and over sampling
def over_under_sample(X_train, y_train, Under=True, Over=True):
    """
    Input: training features and target
    Output: under/oversampled datasets
    """
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42)

    if Under and Over:
        X_train_under, y_train_under = rus.fit_resample (X_train, y_train)
        X_train_over, y_train_over = ros.fit_resample (X_train, y_train)
        return X_train_under, y_train_under, X_train_over, y_train_over
    elif Under:
        X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
        return X_train_under, y_train_under
    else:
        X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
		
		
# Train data preprocessing 
# Input : filename(this is the name of file for training data
# Output : Function will return the value after preprocessing and sampling (X_under, y_under, X_over, y_over)
#          In the output X_under and y_under : Are final X and Y post undersampling that will be used for training model , and other pair is for oversampling
#          but in our case Undersampling gave better results
def traindata_preprocessing(filename):
    # Data loaded to dataframe
    df_raw=pd.read_csv(filename)
    df_raw.info()
    # Null values are dropped in this case it is from Credit_product column, decision for dropping value was taken after detailed EDA, refer PPT file and Solution_withRf_scaled_tuned_ffill_bfill notebook for more info
    df=df_raw.dropna(axis=0)
    df.info()
    
    # Checking for duplicate value in data
    df.duplicated().value_counts()
    
    # Creating dummies for categorical values
    dummies_Gender=pd.get_dummies(df['Gender'])
    dummies_Region=pd.get_dummies(df['Region_Code'])
    dummies_Occupation=pd.get_dummies(df['Occupation'])
    dummies_Channel=pd.get_dummies(df['Channel_Code'])
    dummies_CreditProduct=pd.get_dummies(df['Credit_Product'])
    dummies_IsActive=pd.get_dummies(df['Is_Active'])
    
    # Appending dummies to main dataframe
    df['Gender_Male']=dummies_Gender['Male'].copy()
    df['Gender_Female']=dummies_Gender['Female'].copy()
    df[['RG250', 'RG251', 'RG252', 'RG253', 'RG254', 'RG255', 'RG256', 'RG257',
       'RG258', 'RG259', 'RG260', 'RG261', 'RG262', 'RG263', 'RG264', 'RG265',
       'RG266', 'RG267', 'RG268', 'RG269', 'RG270', 'RG271', 'RG272', 'RG273',
       'RG274', 'RG275', 'RG276', 'RG277', 'RG278', 'RG279', 'RG280', 'RG281',
       'RG282', 'RG283', 'RG284']]=dummies_Region[['RG250', 'RG251', 'RG252', 'RG253', 'RG254', 'RG255', 'RG256', 'RG257',
       'RG258', 'RG259', 'RG260', 'RG261', 'RG262', 'RG263', 'RG264', 'RG265',
       'RG266', 'RG267', 'RG268', 'RG269', 'RG270', 'RG271', 'RG272', 'RG273',
       'RG274', 'RG275', 'RG276', 'RG277', 'RG278', 'RG279', 'RG280', 'RG281',
       'RG282', 'RG283', 'RG284']]
    df['Occupation_Entrepreneur']=dummies_Occupation['Entrepreneur'].copy()
    df['Occupation_Other']=dummies_Occupation['Other'].copy()
    df['Occupation_Salaried']=dummies_Occupation['Salaried'].copy()
    df['Occupation_Self_Employed']=dummies_Occupation['Self_Employed'].copy()
    df['Channel_X1']=dummies_Channel['X1'].copy()
    df['Channel_X2']=dummies_Channel['X2'].copy()
    df['Channel_X3']=dummies_Channel['X3'].copy()
    df['Channel_X4']=dummies_Channel['X4'].copy()
    df['CreditProduct_Yes']=dummies_CreditProduct['Yes'].copy()
    df['CreditProduct_No']=dummies_CreditProduct['No'].copy()
    df['IsActive_Yes']=dummies_IsActive['Yes'].copy()
    df['IsActive_No']=dummies_IsActive['No'].copy()
    
    # Post appending dummies created to main dataframe , drop the categorical value columns
    df=df.drop(['Gender','Region_Code','Occupation','Channel_Code','Credit_Product','Is_Active'],axis=1)
    
    
    df_used=df[['Age', 'Vintage', 'Avg_Account_Balance', 'Is_Lead', 'Gender_Male',
       'Gender_Female', 'RG250', 'RG251', 'RG252', 'RG253', 'RG254', 'RG255',
       'RG256', 'RG257', 'RG258', 'RG259', 'RG260', 'RG261', 'RG262', 'RG263',
       'RG264', 'RG265', 'RG266', 'RG267', 'RG268', 'RG269', 'RG270', 'RG271',
       'RG272', 'RG273', 'RG274', 'RG275', 'RG276', 'RG277', 'RG278', 'RG279',
       'RG280', 'RG281', 'RG282', 'RG283', 'RG284', 'Occupation_Entrepreneur',
       'Occupation_Other', 'Occupation_Salaried', 'Occupation_Self_Employed',
       'Channel_X1', 'Channel_X2', 'Channel_X3', 'Channel_X4',
       'CreditProduct_Yes', 'CreditProduct_No', 'IsActive_Yes', 'IsActive_No']]
    columns=['Age', 'Vintage', 'Avg_Account_Balance', 'Is_Lead', 'Gender_Male',
       'Gender_Female', 'RG250', 'RG251', 'RG252', 'RG253', 'RG254', 'RG255',
       'RG256', 'RG257', 'RG258', 'RG259', 'RG260', 'RG261', 'RG262', 'RG263',
       'RG264', 'RG265', 'RG266', 'RG267', 'RG268', 'RG269', 'RG270', 'RG271',
       'RG272', 'RG273', 'RG274', 'RG275', 'RG276', 'RG277', 'RG278', 'RG279',
       'RG280', 'RG281', 'RG282', 'RG283', 'RG284', 'Occupation_Entrepreneur',
       'Occupation_Other', 'Occupation_Salaried', 'Occupation_Self_Employed',
       'Channel_X1', 'Channel_X2', 'Channel_X3', 'Channel_X4',
       'CreditProduct_Yes', 'CreditProduct_No', 'IsActive_Yes', 'IsActive_No']
    
    #Checking for skewness of data
    for i in columns:
        print("Skewness of "+i+" is :",skew(df_used[i]))
    
    # Handling skewness by applying log transformation, reason for chossing log transformation please refer ppt file and detailed jupyter notebook
    df_used['Vintage']=np.log(df_used['Vintage'])
    df_used['Age']=np.log(df_used['Age'])
    df_used['Avg_Account_Balance']=np.log(df_used['Avg_Account_Balance'])
    
    
    X=df_used[['Age', 'Vintage', 'Avg_Account_Balance', 'RG250',
        'RG252', 'RG254', 'RG256',
       'RG261',  'RG264',  'RG268', 'RG270', 'RG274',
       'RG283', 'RG284', 'Occupation_Entrepreneur', 
       'Occupation_Salaried', 'Occupation_Self_Employed', 'Channel_X1',
       'Channel_X2', 'Channel_X3', 'CreditProduct_Yes',
       'CreditProduct_No', 'IsActive_Yes', 'IsActive_No']]
    Y=df_used[['Is_Lead']]
    
    # Scaling is performed on X and Y so that model can easily learn from result of Scaling 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Creating 2 pairs of X and Y , one is after applying undersampling and one after applying oversampling
    X_under, y_under, X_over, y_over = over_under_sample(X_scaled, np.ravel(Y), Under=True, Over=True)
    return X_under, y_under, X_over, y_over
	

# 	testdata_preprocessing: Function  to preprocess test data , before using it as a input to model for prediction
# Input : filename(this is the name of file for test data
# Output : Function will return the value after preprocessing and scaling (X_test_scaled2)
def testdata_preprocessing(filename):
    df_test=pd.read_csv(filename)
    df_test['Credit_Product'].replace(np.nan,'Yes',inplace=True)
    df_test.info()
    
    # Creating dummies for categorical data
    dummies_Gender=pd.get_dummies(df_test['Gender'])
    dummies_Region=pd.get_dummies(df_test['Region_Code'])
    dummies_Occupation=pd.get_dummies(df_test['Occupation'])
    dummies_Channel=pd.get_dummies(df_test['Channel_Code'])
    dummies_CreditProduct=pd.get_dummies(df_test['Credit_Product'])
    dummies_IsActive=pd.get_dummies(df_test['Is_Active'])
    
    # Appending dummies to dataframe
    df_test['Gender_Male']=dummies_Gender['Male']
    df_test['Gender_Female']=dummies_Gender['Female']
    df_test[['RG250', 'RG251', 'RG252', 'RG253', 'RG254', 'RG255', 'RG256', 'RG257',
       'RG258', 'RG259', 'RG260', 'RG261', 'RG262', 'RG263', 'RG264', 'RG265',
       'RG266', 'RG267', 'RG268', 'RG269', 'RG270', 'RG271', 'RG272', 'RG273',
       'RG274', 'RG275', 'RG276', 'RG277', 'RG278', 'RG279', 'RG280', 'RG281',
       'RG282', 'RG283', 'RG284']]=dummies_Region[['RG250', 'RG251', 'RG252', 'RG253', 'RG254', 'RG255', 'RG256', 'RG257',
       'RG258', 'RG259', 'RG260', 'RG261', 'RG262', 'RG263', 'RG264', 'RG265',
       'RG266', 'RG267', 'RG268', 'RG269', 'RG270', 'RG271', 'RG272', 'RG273',
       'RG274', 'RG275', 'RG276', 'RG277', 'RG278', 'RG279', 'RG280', 'RG281',
       'RG282', 'RG283', 'RG284']]
    df_test['Occupation_Entrepreneur']=dummies_Occupation['Entrepreneur']
    df_test['Occupation_Other']=dummies_Occupation['Other']
    df_test['Occupation_Salaried']=dummies_Occupation['Salaried']
    df_test['Occupation_Self_Employed']=dummies_Occupation['Self_Employed']
    df_test['Channel_X1']=dummies_Channel['X1']
    df_test['Channel_X2']=dummies_Channel['X2']
    df_test['Channel_X3']=dummies_Channel['X3']
    df_test['Channel_X4']=dummies_Channel['X4']
    df_test['CreditProduct_Yes']=dummies_CreditProduct['Yes']
    df_test['CreditProduct_No']=dummies_CreditProduct['No']
    df_test['IsActive_Yes']=dummies_IsActive['Yes']
    df_test['IsActive_No']=dummies_IsActive['No']
    df_test.info()
    df_test=df_test.drop(['Gender','Region_Code','Occupation','Channel_Code','Credit_Product','Is_Active'],axis=1)
    
    # Performing log transformation to handle skewness
    df_test['Vintage']=np.log(df_test['Vintage'])
    df_test['Age']=np.log(df_test['Age'])
    df_test['Avg_Account_Balance']=np.log(df_test['Avg_Account_Balance'])
    
    # Creating set of data post feature importance, for checking approcah for feature importance refer detailed jupyter book
    test_features2=df_test[[ 'Age', 'Vintage', 'Avg_Account_Balance', 'RG250',
        'RG252', 'RG254', 'RG256',
       'RG261',  'RG264',  'RG268', 'RG270', 'RG274',
       'RG283', 'RG284', 'Occupation_Entrepreneur', 
       'Occupation_Salaried', 'Occupation_Self_Employed', 'Channel_X1',
       'Channel_X2', 'Channel_X3', 'CreditProduct_Yes',
       'CreditProduct_No', 'IsActive_Yes', 'IsActive_No']]
    scaler = StandardScaler()
    X_test_scaled2 = scaler.fit_transform(test_features2)
    return X_test_scaled2,df_test

# modelling : This function will take X and Y as input and train the model with same data
# Input : X_train and Y_train (this will be used to train the model)
# Output : Filename ( this is the name of the file for the pickled model
def modelling(X_train,Y_train):
    rf_tuned_under= RandomForestClassifier(n_estimators= 200,
                                           min_samples_split= 2,
                                           min_samples_leaf= 4,
                                           max_depth=21,
                                           bootstrap=True)
    cat_model = CatBoostClassifier(n_estimators=20000, 
                  depth= 4, 
                  learning_rate=0.023, 
                  colsample_bylevel=0.655, 
                  bagging_temperature=0.921, 
                l2_leaf_reg=10.133)
    lgb_model = LGBMClassifier(learning_rate= 0.045, 
             n_estimators= 20000, 
             max_bin= 94,
             num_leaves= 10, 
             max_depth=27, 
             reg_alpha= 8.457, 
             reg_lambda=6.853, 
             subsample=0.749)
    eclf1 = VotingClassifier(estimators=[('lr', rf_tuned_under), ('rf', cat_model), ('gnb', lgb_model)], voting='soft')
    eclf1.fit(X_train,Y_train)
    # save the model to disk
    filename = 'ModelCooked.sav'
    pickle.dump(eclf1, open(filename, 'wb'))
    return filename

# predict : This function will take pickled model file name and store the output result into a csv file
def predict(filename,X_test,df_test):
    eclf1 = pickle.load(open(filename, 'rb'))
    df_test['Is_Lead_rf_tune']=eclf1.predict(X_test)
    df_output1=pd.DataFrame()
    df_output1[['ID','Is_Lead']]=df_test[['ID','Is_Lead_rf_tune']]
    df_output1.to_csv('Output_modular.csv',index=False)

# main: main function containing call to other functions     
def main():
    X_under, y_under, X_over, y_over=traindata_preprocessing('train_s3TEQDk.csv')
    filename=modelling(X_under,y_under)
    X_test_scaled2,df_test=testdata_preprocessing('test_mSzZ8RL.csv')
    predict(filename,X_test_scaled2,df_test)

if __name__ == "__main__":
    main()
