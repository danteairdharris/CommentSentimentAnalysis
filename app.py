# for app deployment 
import streamlit as st
st.set_page_config(layout="wide")

# for data management
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# for model testing in app w/ tensorflow
# import tensorflow as tf
# import tensorflow_hub
# import tensorflow_text

# @st.cache
# def load_use(url):
#     return tensorflow_hub.load(url)


# for model testing in app w/ sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_sent(pred):
    if np.argmax(pred) == 0:
        return 'negative'
    else: 
        return 'positive'
    
# def transform_predict(text):
#     text_arr = []
#     emb = use(text)
#     text_emb = tf.reshape(emb, [-1]).numpy()
#     text_arr.append(text_emb) 
#     text_arr = np.array(text_arr)
#     return get_sent(model.predict(text_arr))

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

@st.cache
def sk_test(text):
    text_arr = [text]
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(fit_vec)
    test_vec = vectorizer.transform(text_arr)
    pred = model.predict(test_vec)
    return str(pred[0])

@st.cache
def vectorize_df(df):
    all_comments = df['Comment'].str.cat()
    corpus = [all_comments]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,1), max_df = 1, min_df = .0001)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    dense = X.todense()
    denselist = dense.tolist()
    # tf-idf matrix
    tfidf_df = pd.DataFrame(denselist, columns=feature_names)
    
    data = tfidf_df.transpose()
    data.columns = ['comments']
    return data



def white_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
        return("hsl(0,0%,100%)")
    
@st.cache
def wordcloud(vec_df):
    # set the wordcloud background color to white
    # set max_words to 100
    # set width and height to higher quality, 2000 x 1000
    wordcloud = WordCloud(background_color="black", width=2000, height=1000, max_words=100).generate_from_frequencies(vec_df['comments'])
    # set the word color to black
    wordcloud.recolor(color_func = white_color_func)
    return wordcloud

    
## tf model 
# use = load_use("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
# model = tf.keras.models.load_model('./Model/saved_model/sentiment_analysis_model')

## sklearn model
model = load_data('./Model/logreg_model.pickle')
fit_vec = load_data('./Model/fit_vector.pickle')

data = pd.read_csv('./movie_trailer_comments_sentiment.csv')
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
    #### The basic Tensorflow model I built uses data from IMDB dataset with 50k samples to classify movie review sentiment as positive or negative. It was evaluated at 84% accuracy. Here the model is being used to classify comment sentiment from youtube trailers of upcoming movies. Compare the model's performance against a model built from sklearn ML library evaluated at 90% accuracy.
""")
    col2_container.text('\n\n')
    col2_container.write("""
    ### Example:
    * index: 41,  Title: DONT BREATH 2
    * Comment: 'Somehow, I think the continuation of the previous story could have been a better idea.'  
    * TensorFlow model : positive,  SKLearn model : negative
""")
#     col2_container.markdown("![Alt Text](https://github.com/danteairdharris/CommentSentimentAnalysis/blob/master/app_demo.gif)")
    col2_container.image('./app_demo.gif')
    col2_container.text('\n')
    col2_container.write('Test the model by inputing some text below and having it classified.')
    
    text = col2_container.text_input('Input')
    
#     sklearn model test
    if text:
        col2_container.text('SKLearn classification: ' + sk_test(text))

    
#     tensorflow model test
#     if text:
#         text_arr = []
#         emb = use(text)
#         text_emb = tf.reshape(emb, [-1]).numpy()
#         text_arr.append(text_emb)

#         text_arr = np.array(text_arr)
#         col2_container.text(pred_sent(text_arr))


data_overview = col1_container_top.beta_expander('Dynamic Data Overview')
with data_overview:
    
    ## plot data
    dfcount = figure(df)
    dfcount = dfcount.reset_index()
    fig, ax = plt.subplots()
    width = 0.35
    ax.barh(dfcount['Title'], dfcount['Positive'], width, label='postive')
    ax.barh(dfcount['Title'], dfcount['Negative'], width, align='edge', label='negative')
    
    # Overall Sentiment   
    pos = dfcount['Positive'].sum()
    neg = dfcount['Negative'].sum()
    ratio = pos / neg
    if pos > neg:
        overall = 'Positive'
    elif pos < neg:
        overall = 'Negative'
    else:
        overall = 'Neutral'

    data_overview.write('Overall comment sentiment for selected dataframe: ')
    data_overview.text(overall + ' with a %.2f to 1 ratio' % ratio)
    
  
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
    data_overview.pyplot(fig)
    
    
    data_overview.write('Wordcloud for selected Dataframe: ')
    ## mount wordcloud
    # concatenate all comments into a corpus for vectorization
    vec_df = vectorize_df(df)
    wordcloud = wordcloud(vec_df)
    # set the figsize
    plt.figure(figsize=[15,10])
    ## plot wordcloud
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(wordcloud)
    plt.savefig('cloud.png', bbox_inches='tight',pad_inches = 0)
    data_overview.image('cloud.png', width=875)
    
## render dataframe
col1_container_top.dataframe(df)

## body spacing
col1_container_mid.text('\n')
col1_container_mid.text('\n')

## if all movies are currently selected
# if len(df) == 988:
   

# else :
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


    
    


            