import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from GB (GBvideos.csv and GB_category_id.json) and answer the questions.
"""

def Q1():
    """
        1. How many rows are there in the GBvideos.csv after removing duplications?
        - To access 'GBvideos.csv', use the path '/data/GBvideos.csv'.
    """
    
    # TODO: Paste your code here
    df=pd.read_csv("/data/GBvideos.csv")
    return (df.drop_duplicates().shape[0])
    

def Q2(vdo_df):
    '''
        2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO: Paste your code here

    return (vdo_df[vdo_df.dislikes>vdo_df.likes].title.nunique())

def Q3(vdo_df):
    '''
        3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    '''
    # TODO: Paste your code here
    return (vdo_df[(vdo_df.trending_date == '18.22.01') & (vdo_df.comment_count > 10000)].title.count())
    

def Q4(vdo_df):
    '''
        4. Which trending date that has the minimum average number of comments per VDO?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO:  Paste your code here

    return (vdo_df.groupby('trending_date')['comment_count'].mean().idxmin())
    

def Q5(vdo_df):
    '''
        5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - You must load the additional data from 'GB_category_id.json' into memory before executing any operations.
            - To access 'GB_category_id.json', use the path '/data/GB_category_id.json'.
    '''
    # TODO:  Paste your code here

    with open('US_category_id.json') as c:
        cat = json.load(c)
    cat_list = []
    for d in cat['items']:
        cat_list.append((int(d['id']), d['snippet']['title']))
    cat_df = pd.DataFrame(cat_list, columns=['id', 'category'])
    print(cat_df.head(3))
    vdo_df_withcat = vdo_df.merge(cat_df, left_on='category_id', right_on='id')
    df = vdo_df_withcat.groupby(["trending_date","category"])["views"].sum().reset_index(name='sum')
    print(df.head(3))
    sport = df[df['category'] == 'Sports'].reset_index()
    comedy = df[df['category'] == 'Comedy'].reset_index()
    print(sport.head(3))
    print(type(sport.category))
    return (sport['sum']>comedy['sum']).sum()
df = pd.read_csv('USvideos.csv')
print(Q5(df))