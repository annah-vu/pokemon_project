'''
Welcome to the explore.py, your one stop shop for all things visualize and stats testing!
'''

#imports
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from scipy.stats import f_oneway

def get_pokemon_heatmap(df):
    '''returns a beautiful heatmap with correlations according to simplified catch_rates'''
    plt.figure(figsize=(8,12))
    poke_heatmap = sns.heatmap(df.corr()[['simplified_catch_rate']].sort_values(by='simplified_catch_rate', ascending=False), vmin=-.5, vmax=.5, annot=True,cmap='seismic')
    poke_heatmap.set_title('Features Correlated with Catch Rates')
    return poke_heatmap



def explore_univariate(df, variable):
    '''
    explore_univariate will take in a dataframe, and one feature or variable. It graphs a box plot and a distribution 
    of the single variable.
    '''
    #set figure size, font for axis ticks, and turns off gridlines.
    plt.figure(figsize=(30,10))
    sns.set(font_scale = 2)
    sns.set_style("whitegrid", {'axes.grid' : False})
    
    # boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=variable, data=df)
    plt.xlabel('')
    plt.title('Box Plot', fontsize=30)
    
    # distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=variable, element='step', kde=True, color='blue')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Distribution', fontsize=30)
    
    #title
    plt.suptitle(f'{variable}', fontsize = 45)
    plt.tight_layout()
    plt.show()

def histplot(df, variable, target):
    '''
    takes in a df, variable, and target and creates a histogram with target as the hue
    '''
    plt.figure(figsize=(12,8))
    sns.histplot(data=df, x=variable, hue=target, multiple='stack')
    plt.show()
    
def count_and_histplots(df, variable, target):
    '''
    Takes in a dataframe, variable column, and target column and creates a 
    ountplot and histplot side by side
    '''
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=(16,7))
    sns.countplot(data=df, hue=variable, x=target, ax=ax1)
    sns.histplot(data=df, x=variable, hue=target, multiple='stack', ax=ax2)
    plt.show()

def scatterplot(train, x, y):
    '''
    scatterplot takes in a dataframe, x variable, y variable, and makes a scatterplot with 
    simplified_catch_rate as the hue
    '''
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=x,y=y,data=train,hue='simplified_catch_rate', palette='brg_r')
    plt.title(f'{x} and {y}', fontsize = 20)
    #plt.legend(loc="upper center", bbox_to_anchor=(1, 1), ncol=1) #hm
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.show()

def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
    
def inertia(X):  
    '''
    returns inertia graph so we can decide how many clusters to use
    '''
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
    

def make_cluster(X_train_scaled, X, n, variable_1, variable_2):
    '''
    makes and graphs cluster based off X_train_scaled, selected features, number of clusters, and 2 variables
    '''
    X_train_scaled, X_scaled, scaler, kmeans, centroids = create_cluster(X_train_scaled, X, n)
    create_scatter_plot(variable_1, variable_2,X_train_scaled,kmeans, X_scaled, scaler)
    print(f'Clusters based on {variable_1} and {variable_2} of Pokemon')
    
def chi2test(df, x, y):
    '''
    performs a chi2 test for independence by taking in a dataframe, and 2 variables
    uses alpha of 0.05
    '''
    a = 0.05 #a for alpha 

    observed = pd.crosstab(df[x], df[y], margins = True)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < a:
        print(f'Reject null hypothesis. There is evidence to suggest {x} and {y} are not independent.\n p-value is {p}')
    else:
        print(f"Fail to reject the null hypothesis. There is not sufficient evidence to reject independence of {x} and {y}.\n p-value is {p}")
        
def anova_test(df, x):
    '''
    performs an ANOVA test based on inputted dataframe and variable with the 5 simplified catch rates.
    It returns the p value and whether to reject or fail to reject the null hypothesis that there was independence
    '''
    a=0.05
    catch1 = df[df['simplified_catch_rate']==1][x]
    catch2 = df[df['simplified_catch_rate']==2][x]
    catch3 = df[df['simplified_catch_rate']==3][x]
    catch4 = df[df['simplified_catch_rate']==4][x]
    catch5 = df[df['simplified_catch_rate']==5][x]
    f, p = f_oneway(catch1,catch2,catch3,catch4,catch5)
    print(f'p value is {p} for simplified_catch_rate and {x}')
    if p>a:
        print('Aw man! Fail to reject the null hypothesis! No significant difference in means between the catch rate groups.')
    else:
        print('Got em! There\'s evidence to suggest at least 2 of these groups have different means for catch rate!')
        
        
def plot_categorical_and_continuous_vars(df, cat_vars, quant_vars):
    '''takes in a dataframe as input, with a discrete, and continuous variable and returns 
    and barplot, swarm plot, boxplot'''
    plt.figure(figsize=(15,7))
    sns.barplot(data=df, y=quant_vars, x=cat_vars,hue='simplified_catch_rate')
    plt.legend(loc="upper center", bbox_to_anchor=(1, 1), ncol=1)
    plt.show()
    plt.figure(figsize=(15,7))
    sns.swarmplot(data=df, y=quant_vars, x=cat_vars, hue='simplified_catch_rate')
    plt.legend(loc="upper center", bbox_to_anchor=(1, 1), ncol=1)
    plt.show()
    plt.figure(figsize=(15,7))
    sns.boxplot(data=df, y=quant_vars, x=cat_vars, hue='simplified_catch_rate')
    plt.legend(loc="upper center", bbox_to_anchor=(1, 1), ncol=1)
    
def selectkbest(X_train_scaled, y_train, n):
    '''
    selectkbest takes in X_train scaled, y_train, and a desired number of features and returns 
    the selected features to be used in modeling
    '''
    f_selector = SelectKBest(k=n)
    f_selector.fit(X_train_scaled, y_train)
    f_support = f_selector.get_support()
    f_feature = X_train_scaled.loc[:,f_support].columns.tolist()
    print(str(len(f_feature)), 'selected features')
    print(f_feature)
    return f_feature

def categorical_bar(data, x):
    '''
    takes in a dataframe and a variable, and plots a countplot 
    '''
    sns.countplot(data=data, x=x)
    plt.show()
    return pd.crosstab(index=data[x], columns='count')