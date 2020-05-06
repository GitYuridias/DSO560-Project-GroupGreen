#!/usr/bin/env python
# coding: utf-8

# In[5]:
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import en_core_web_sm
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings("ignore")
nlp = en_core_web_sm.load()
from nltk.stem import PorterStemmer

def stem_text(text):
    porter=PorterStemmer()
    tokens = text.split()
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def outfit_recommendation(input_value):
    outfit = pd.read_csv('https://dso-560-nlp-text-analytics.s3.amazonaws.com/outfit_combinations.csv')
    full_data = pd.read_csv('https://dso-560-nlp-text-analytics.s3.amazonaws.com/Full+data.csv')
    
    full_data = full_data.loc[:,['product_id', 'description']]
    
    combined = pd.merge(outfit,
                    full_data,
                    on = 'product_id')
    
    input_value = input_value.upper()
    if combined.loc[combined['product_id']==input_value,'product_id'].any() == input_value:
        input_descr = combined.loc[combined['product_id']==input_value,'description'].values[0]
    else:
        input_descr = input_value
    
    
    col = ['outfit_item_type', 'brand', 'product_full_name', 'description']

    for columns in col:
        combined[columns] = combined[columns].str.lower() 
    combined.sort_values(by='outfit_id', inplace = True)
    combined.reset_index(drop = True, inplace = True)

    stopwords_gensim = list(STOPWORDS)
    stopwords_NLTK = list(stopwords.words("english"))
    stopwords_combined = list(set(stopwords_gensim+stopwords_NLTK)) #to remove duplicates
    negatives = ['not','nor','no','neither', 'never', 'bottom', 'top'] #took out the negative words for a more accurate analysis
    stopwords_combined = list(filter(lambda x: x not in negatives, stopwords_combined))
    stopwords_combined.sort()
    stopwords_expression = '|'.join(stopwords_combined)
    stopwords_pattern = f'({stopwords_expression})'
    
    combined['description'] = combined['description'].astype(str)
    combined['description'] = combined['description'].str.replace(r'[^\w\s]',' ')
    combined['description'] = combined['description'].str.replace(rf'\b{stopwords_pattern}\b','')
    combined['description'] = combined['description'].apply(stem_text)

    description = list(combined.description) #making the description into a list
    description = [str(i) for i in description] #making all the elements into string
    description = list(set(description))
    description_vectors = []
    for i in description:
        temp_description = nlp(i)
        description_vectors.append(temp_description.vector)
        
    input_clean = input_descr.lower() 
    input_clean= re.sub(r'[^\w\s]',' ',input_clean)
    input_clean = re.sub(rf'\b{stopwords_pattern}\b','',input_clean)
    input_clean = stem_text(input_clean)


    description.append(input_clean)

    input_clean_vectors = []
    temp_input_clean = nlp(input_clean)
    description_vectors.append(temp_input_clean.vector)
    
    
    vector_df =pd.DataFrame(description_vectors)
    vector_df["description"] = description

    vector_df.set_index("description", inplace=True)

    from sklearn.metrics.pairwise import cosine_similarity

    similarities = pd.DataFrame(cosine_similarity(vector_df.values), columns=description, index=description)

    top_similarities = similarities.unstack().reset_index()
    top_similarities.columns = ["input_clean", "original_description", "similarity"]
    top_similarities = top_similarities.sort_values(by="similarity", ascending=False)
    top_similarities = top_similarities[top_similarities["similarity"] < .9999]
    
    match = top_similarities.loc[top_similarities['input_clean']== input_clean,:].reset_index(drop = True)
    
    input_clean_item = match.loc[0,'original_description']
    output = combined.loc[combined.description == input_clean_item,:]

    for i in range(1,len(match)):
        match_item = match.loc[i,'original_description']
        match_combined = combined.loc[combined.description == match_item,:]
        output = pd.concat([output, match_combined])
        if output.outfit_item_type.nunique() >= 3:
            break
        else:
            continue

    output.drop_duplicates(subset= ['outfit_item_type'], keep = 'first', inplace=True)
    output.drop_duplicates(subset= ['product_id'], keep = 'first', inplace=True)
    output.reset_index(drop = True, inplace = True)
    
    product_list = []
    for i in range(0, len(output)):
        product_list.append(output.loc[i,'product_id'])
    
    data = pd.merge(outfit,
                    full_data,
                    on = 'product_id')

    recommendation_output = []


    if data.loc[data['product_id']==input_value,'product_id'].any() == input_value:
  
        recommendation_output = data.loc[data['product_id']==input_value,:].head(1)

        recommendation = []
        for i in product_list:
            recommendation.append(data.loc[data.product_id == i,:])    

        recommendation = pd.concat(recommendation)
        recommendation.drop_duplicates(keep = 'first', subset= ['product_id'], inplace=True)
        recommendation.reset_index(drop = True, inplace = True)
        recommendation_output = recommendation_output.append(recommendation)

    else:    
        recommendation_output = []
        for i in product_list:
            recommendation_output.append(data.loc[data.product_id == i,:])

        recommendation_output = pd.concat(recommendation_output)
        recommendation_output.drop_duplicates(keep = 'first', subset= ['product_id'], inplace=True)
    recommendation_output.reset_index(drop = True, inplace = True)
    recommendation_output = recommendation_output.iloc[:,1:6]
    
    return recommendation_output

