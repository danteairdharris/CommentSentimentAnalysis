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

def sort_by_title(df, select, options):  
    if select == 'All':
        return all_df
    else:
        for op in options:
                if select == op:
                    df = df[df.Title == select]
                    df.reset_index(inplace=True)
                    df.drop('index', axis=1, inplace=True)
                    return df

def sort_by_sentiment(df, select, options):
    if select == options[0]:
        return df
    elif select == options[1]:
        df = df[df.Comment_Sentiment == options[1]]
    elif select == options[2]:
        df = df[df.Comment_Sentiment == options[2]]
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    return df
    
def get_i(df, start, end):
    return df.iloc[start:end]
    
def binary(string, check):
    if string == check:
        return 1
    return 0

@st.cache
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
data = pd.read_csv('./Data/movie_trailer_comments_sentiment.csv')
all_df = data.drop(['Unnamed: 0', 'index'], axis=1)
df = all_df
col1, col2 = st.beta_columns([2,1])

st.sidebar.write("""
    ### Filter Comments by Features
""")

# For Filtering by Movie Title
titles_arr = ['All']
titles = df['Title'].unique()
for title in titles:
    titles_arr.append(title)
select1 = st.sidebar.selectbox('Movie', options=titles_arr, index=0)
df = sort_by_title(df, select1, titles_arr)


# For filtering by Comment Sentiment 
sentiments = ['All', 'positive', 'negative']
select2 = st.sidebar.selectbox('Sentiment', options=sentiments)
df = sort_by_sentiment(df, select2, sentiments)


# For filtering by datframe index
st.sidebar.text('\n\n')
st.sidebar.write('### Splicing by index')
start = st.sidebar.text_input('Start')
end = st.sidebar.text_input('End')
if start and end:
    start = int(start)
    end = int(end)
    if start < 0 or start > len(df) or end < 0 or end <= start or end > len(df):
        st.sidebar.text('Some index out of bounds.')
        st.sidebar.text('Check that index are chronological and within the dataframe.')
    else:
        df = get_i(df, start, end)

        
with col1: 
    col1_container_top = st.beta_container()
    col1_container_mid = st.beta_container()
    col1_container_bot = st.beta_container()
    col1_container_top.write("""
         ## Movie Trailer Youtube Comments Data
    """)
    
with col2:
    col2_container = st.beta_container()
    col2_container.text('\n\n')
    col2_container.write("""
    ### About Model
    #### The basic Tensorflow model I built uses data from IMDB dataset with 50k samples to classify movie review sentiment as positive or negative. It was evaluated at 84% accuracy. Here the model is being used to classify comment sentiment from youtube trailers of upcoming movies.
""")
    col2_container.text('\n\n')
    col2_container.write("""
    ### Example:
    * 1) 'The last movie was really good. I Can't wait for the sequel, so excited.'  (positive) 
    * 2) 'This movie doesnt look that interesting. I'll pass.'  (negative) 
""")
    col2_container.text('\n')
    col2_container.write('Test the model by inputing some text below and having it classified.')
    
    text = col2_container.text_input('Input')
    if text:
        text_arr = []
        emb = use(text)
        text_emb = tf.reshape(emb, [-1]).numpy()
        text_arr.append(text_emb)

        text_arr = np.array(text_arr)
        col2_container.text(pred_sent(text_arr))




## render dataframe
col1_container_top.dataframe(df)

## body spacing
col1_container_mid.text('\n')
col1_container_mid.text('\n')

## if all movies are currently selected
if len(df) == 988:
    ## plot data
    dfcount = figure(df)
    dfcount = dfcount.reset_index()
    fig, ax = plt.subplots()
    width = 0.35
    ax.barh(dfcount['Title'], dfcount['Positive'], width, label='postive')
    ax.barh(dfcount['Title'], dfcount['Negative'], width, align='edge', label='negative')

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
    plt.setp(plt.title('Positive(blue) and Negative(orange) Reviews by Movie Title', color='w'))
    col1_container_bot.pyplot(fig)

else :
    df_table = df
    ## else render comment/comments
    col1_container_bot.text('Filtering... ')
    no_comments = len(df_table)
    if len(df_table) == 988:
        movie_title = select1
    else:
        movie_title = df_table.Title.unique().tolist()
    sentiment = select2
    col1_container_bot.text('# of Comments: ' + str(no_comments))
    col1_container_bot.text(movie_title)
    col1_container_bot.text('Sentiment: ' + sentiment)
    col1_container_bot.write(""" 
        ## Comments
    """)
    col1_container_bot.table(df_table.drop(['Title', 'Comment Length', 'Language'], axis=1))


    
    


            