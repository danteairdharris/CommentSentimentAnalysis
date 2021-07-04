# for app deployment 
import streamlit as st
st.set_page_config(layout="wide")

# for data management
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub
import tensorflow_text
import numpy as np
import seaborn as sns

@st.cache
def load_use(url):
    return tensorflow_hub.load(url)
     
def get_sent(pred):
    if np.argmax(pred) == 0:
        return 'negative'
    else: 
        return 'positive'
    
def transform_predict(text):
    text_arr = []
    emb = use(text)
    text_emb = tf.reshape(emb, [-1]).numpy()
    text_arr.append(text_emb) 
    text_arr = np.array(text_arr)
    return get_sent(model.predict(text_arr))

def sort_by_language(select, options):
    if select == options[0]:
        return all_df
    elif select == options[1]:
        return df[df.Language == options[1]]

def sort_by_title(df, select, options):
    for s in select:
        if s not in df['Title']:
            df = pd.concat([df, all_df[all_df.Title == s]])
            break
            
    for op in options:
        if op not in select:
            df = df[df.Title != op]
    return df

def sort_by_sentiment(df, select, options):
    if select == options[0]:
        return df
    elif select == options[1]:
        return df[df.Comment_Sentiment == options[1]]
    elif select == options[2]:
        return df[df.Comment_Sentiment == options[2]]
    
def binary(string, check):
    if string == check:
        return 1
    return 0
    
def figure(df):
    dfcount = df[['Title','Comment_Sentiment']]
    dfcount['Positive'] = dfcount['Comment_Sentiment'].apply(lambda x: binary(x, 'positive'))
    dfcount['Negative'] = dfcount['Comment_Sentiment'].apply(lambda x: binary(x, 'negative'))
    dfcount = dfcount.drop('Comment_Sentiment', axis=1).groupby('Title').sum()
    dfcount = dfcount.sort_values(by=['Positive'], ascending=False)
    return dfcount

def get_sent(pred):
    if np.argmax(pred) == 0:
        return 'negative'
    else: 
        return 'positive'
    
def pred_sent(text):
    pred = model.predict(text)
    return get_sent(pred)

    

use = load_use("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
model = tf.keras.models.load_model('./Model/saved_model/sentiment_analysis_model')
all_df = pd.read_csv('./Data/movie_trailer_comments_sentiment.csv')
all_df.drop(['Unnamed: 0', 'index'], axis=1, inplace=True)
df = all_df
col1, col2 = st.beta_columns([2,1])


st.sidebar.write("""
    ### Filter Comments by Features
""")
# For Filtering by Comment Language
langs_arr = ['All', 'en']
select = st.sidebar.selectbox('Language', options=langs_arr)
df = sort_by_language(select, langs_arr)

# For Filtering by Movie Title
titles_arr = []
titles = df['Title'].unique()
for title in titles:
    titles_arr.append(title)
select1 = st.sidebar.multiselect('Movie', options=titles_arr, default=titles_arr)
df = sort_by_title(df, select1, titles_arr)


# For filtering by Comment Sentiment 
sentiments = ['All', 'positive', 'negative']
select2 = st.sidebar.selectbox('Sentiment', options=sentiments)
df = sort_by_sentiment(df, select2, sentiments)


with col1: 
    col1_container = st.beta_container()
    col1_container.write("""
         ## Movie Trailer Youtube Comments Data
    """)
    
with col2:
    col2_container = st.beta_container()
    col2_container.write("""
""")
    text = col2_container.text_input('Input')
    if text:
        text_arr = []
        emb = use(text)
        text_emb = tf.reshape(emb, [-1]).numpy()
        text_arr.append(text_emb)

        text_arr = np.array(text_arr)
        col2_container.text(pred_sent(text_arr))




## render dataframe
col1_container.dataframe(df)

## plot data
dfcount = figure(df)
dfcount = dfcount.reset_index()
fig, ax = plt.subplots()
ax.barh(dfcount['Title'], dfcount['Positive'], 0.35, label='postive')
ax.barh(dfcount['Title'], dfcount['Negative'], 0.35, align='edge', label='negative')

## styling plot
plt.xticks(rotation=90)
ax.set_facecolor('#000000')
fig.patch.set_facecolor('#000000')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
plt.setp(plt.title('Amount of Positive and Negative Reviews by Movie Title', color='w'))
col1_container.pyplot(fig)


    
    


            