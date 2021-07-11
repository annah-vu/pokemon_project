#imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Acquire Functions

def get_pokemon():
    '''
    this function will retrieve the original pokedex.csv file for use, and return as a dataframe
    '''
    df = pd.read_csv('pokedex.csv')
    return df

def prepare_pokemon(df):
    '''
    prepare_pokemon will drop columns we do not intend to use, and fills in nulls
    '''
    
    #drop columns I know I won't be using
    df= df.drop(columns=['Unnamed: 0','german_name','japanese_name'])
    
    #fill pokemon's second type with none if they only have a primary type
    df['type_2'].fillna('None', inplace=True)
    
    #fill in 1 missing weight, this Pokemon does not have a specified weight so I just scaled it proportionatly. 
    df['weight_kg'].fillna(4750, inplace=True)
    
    #Pokemon with missing ability_1 values are basically identical to the entry above them. Use forward fill. 
    df['ability_1'].fillna(method='ffill', inplace=True)
    
    #if they don't have a second ability, put none
    df['ability_2'].fillna('None', inplace=True)
    
    #for pokemon with no hidden abilities, put none
    df['ability_hidden'].fillna('None', inplace=True)
    
    #Pokemon with missing catch rates are basically identical to the entry above them. Use forward fill.
    df['catch_rate'].fillna(method='ffill', inplace=True)
    
    #if there is no base friendship among capture, there is no base friendship at all. Fill with 0.
    df['base_friendship'].fillna(0, inplace=True)
    
    #same for experience
    df['base_experience'].fillna(0, inplace=True)
    
    #None for pokemon with no growth rate
    df['growth_rate'].fillna('None', inplace=True)
    
    #No egg type
    df['egg_type_1'].fillna('None', inplace=True)
    
    #and no second egg type
    df['egg_type_2'].fillna('None', inplace=True)
    
    #create is_genderless column for our Pokemon, 0 if False, 1 if True
    df['is_genderless']= (df['percentage_male'].isnull()==True).astype(int)
    
    #genderless Pokemon have no value in percentage_male, I'll fill with 200 so it's out of the range 0-100.
    df['percentage_male'].fillna(200, inplace=True)
    
    #no egg cycles 
    #you get no egg cycle.
    df['egg_cycles'].fillna(0, inplace=True)
    
    #let's reset the index
    df = df.reset_index(drop=True)
    
    return df

def ready_for_battle(df):
    '''
    ready_for_battle takes in the pokedex.csv and prepares it as in the prepare_pokemon function
    it also replaces inapprorpiate data in some of our ice pokemon, turns object columns into new classifiables ones ending
    in ???_num, and sets up a column called simplified_catch_rate which categorizes certain ranges of catch rates into
    a category between 1 and 5. 1 being the hardest, 5 being the easiest to catch
    '''
    #fill in missing values
    df = prepare_pokemon(df)
    
    #adjust weird values
    df['against_ice'].replace({125:0.25}, inplace=True)
    
    #create abiility_1_num
    ability_list = df.ability_1.value_counts().index.to_list()
    for count, value in enumerate(ability_list):
        df.loc[df['ability_1']==value,'ability_1_num']=count
    
    #create ability_2_num
    ability_2_list = df.ability_2.value_counts().index.to_list()
    for count, value in enumerate(ability_2_list):
        df.loc[df['ability_2']==value,'ability_2_num']=count
    
    #create ability_hidden_num
    ability_hidden_list = df.ability_hidden.value_counts().index.to_list()
    for count, value in enumerate(ability_hidden_list):
        df.loc[df['ability_hidden']==value,'ability_hidden_num']=count
    
    #create status_num
    status_list = df.status.value_counts().index.to_list()
    for count, value in enumerate(status_list):
        df.loc[df['status']==value,'status_num']=count
    
    #create primary_num (for their types)
    primary_list = df.type_1.value_counts().index.to_list()
    for count, value in enumerate(primary_list):
        df.loc[df['type_1']==value,'primary_num']=count
    
    #create secondary_num (for their second type)
    secondary_list = df.type_2.value_counts().index.to_list()
    for count, value in enumerate(secondary_list):
        df.loc[df['type_2']==value,'secondary_num']=count

    #create growth_num 
    growth_list = df.growth_rate.value_counts().index.to_list()
    for count, value in enumerate(growth_list):
        df.loc[df['growth_rate']==value,'growth_num']=count

    #create simplified_catch_rate column  
    df['simplified_catch_rate'] = 0
    
    #for pokemon with a catch rate between 0 and 25, put them in category 1 (the most difficult)
    df['simplified_catch_rate'] = np.where(df['catch_rate'].between(0,25), 1, 0)
    
    #for pokemon with a catch rate between 30 and 80, put them in category 2 (hard)
    df['simplified_catch_rate'] = np.where(df.catch_rate.between(30,80), 2, df['simplified_catch_rate'])
    
    #for pokemon with a catch rate between 90 and 150, put them in category 3 (medium)
    df['simplified_catch_rate'] = np.where(df.catch_rate.between(90,150), 3, df['simplified_catch_rate'])
    
    #for pokemon with a catch rate between 155 and 205, put them in category 4 (easy)
    df['simplified_catch_rate'] = np.where(df.catch_rate.between(155,205), 4, df['simplified_catch_rate'])
    
    #for pokemon with a catch rate between 220 and 255, put them in category 5 (very easy)
    df['simplified_catch_rate'] = np.where(df.catch_rate.between(220,255), 5, df['simplified_catch_rate'])
    # even though the bins don't capture every numeric value, it captures all of the Pokemon. 
    
    #reset index
    df = df.reset_index(drop=True)
    
    return df

def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns train, validate, test sets and also another 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols

def pokemon_split(df):
    '''splitting our data, stratifying simplified catch rates.'''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.simplified_catch_rate)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.simplified_catch_rate)
    return train, validate, test


def split_X_y(train, validate, test, target):
    '''
    Splits train, validate, and test into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def standard_scale_data(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    return X_train_scaled, X_validate_scaled, X_test_scaled