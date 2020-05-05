#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from collections import OrderedDict
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

def stem_text(text):
    porter=PorterStemmer()
    tokens = text.split()
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def import_data():
    full_data = pd.read_csv('https://dso-560-nlp-text-analytics.s3.amazonaws.com/Full+data.csv')
    extra_data = pd.read_csv('https://dso-560-nlp-text-analytics.s3.amazonaws.com/extra_data.csv')
    sme_tag = pd.read_excel('https://dso-560-nlp-text-analytics.s3.amazonaws.com/USC+Product+Attribute+Data+03302020.xlsx')
    additional_tag = pd.read_csv('https://dso-560-nlp-text-analytics.s3.amazonaws.com/usc_additional_tags.csv')
    full_data = full_data.iloc[:,:14]

    clean_df = pd.concat([full_data, extra_data], ignore_index=True)
    clean_df.drop_duplicates(subset= ['product_id'], keep = 'first', inplace=True)
    clean_df = clean_df[list(full_data.columns)]

    train = clean_df.drop(columns=['created_at', 'updated_at', 'deleted_at', 'bc_product_id'])

    for columns in train.columns:
        train[columns] = train[columns].str.lower() 

        
    tag = pd.concat([sme_tag, additional_tag], ignore_index=True)
    for columns in tag.columns:
        tag[columns] = tag[columns].str.lower() 

    
    remove = [' ', '_', '(', ')', '-', ',', '&', '"', '"', '/']
    for i in remove:
        tag['attribute_name'] = tag['attribute_name'].str.replace(i, '')
        tag['attribute_value'] = tag['attribute_value'].str.replace(i, '')

    tag.drop_duplicates(keep='first', inplace=True) 
    tag.reset_index(inplace = True)
    
    train['style'] = ""
    train['occasion'] = ""
    train['category'] = ""
    train['embellishment'] = ""
    
    
    group_tag = tag.groupby('product_id')
    tagged_product = list(group_tag.groups.keys())
    
    
    focus_attribute = ['style', 'occasion', 'category', 'embellishment']

    for x in tagged_product:
        product = group_tag.get_group(x)
        focus = product.loc[product['attribute_name'].isin(focus_attribute)].reset_index()
        for i in range(0,len(focus)):
            if focus.loc[i, 'attribute_name'] == 'style':
                train.loc[train['product_id'] == x, 'style'] += (focus.loc[i, 'attribute_value'] + " ")
            elif focus.loc[i, 'attribute_name'] == 'occasion':
                train.loc[train['product_id'] == x, 'occasion'] += (focus.loc[i, 'attribute_value'] + " ")
            elif focus.loc[i, 'attribute_name'] == 'category':
                train.loc[train['product_id'] == x, 'category'] += (focus.loc[i, 'attribute_value'] + " ")
            elif focus.loc[i, 'attribute_name'] == 'embellishment':
                train.loc[train['product_id'] == x, 'embellishment'] += (focus.loc[i, 'attribute_value'] + " ")
                
    
    for i in focus_attribute:
        train[i] = (train[i].str.split()
                                  .apply(lambda x: OrderedDict.fromkeys(x).keys())
                                  .str.join(' '))
        
        
      

    stopwords_gensim = list(STOPWORDS)
    stopwords_NLTK = list(stopwords.words("english"))
    stopwords_combined = list(set(stopwords_gensim+stopwords_NLTK))
    negatives = ['not','nor','no','neither', 'never'] 
    stopwords_combined = list(filter(lambda x: x not in negatives, stopwords_combined))
    stopwords_combined.sort()
    stopwords_expression = '|'.join(stopwords_combined)
    stopwords_pattern = f'({stopwords_expression})'

    col = ['brand', 'name', 'description', 'brand_category', 'details', 'labels', 'tsv']

    for i in col:
        train[i] = train[i].astype(str)
        train[i] = train[i].str.replace('[^\w\s]',' ')
        train[i] = train[i].str.replace("\n", " ")
        train[i] = train[i].str.replace(rf'\b{stopwords_pattern}\b','')
        train[i] = train[i].apply(stem_text)


    data = train
    data = data.replace('nan',np.NaN)
    data = data.replace('',np.NaN)
        
    
    return data



def labelled_data(data):
    data_label = data.loc[(data['style'].notna()) | (data['occasion'].notna()) |
                          (data['category'].notna()) | (data['embellishment'].notna())]
    data_label.reset_index(inplace = True)

    return data_label


    

if __name__ == '__main__':
    import_data()

