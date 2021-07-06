# [Youtube Comments Text Sentiment Analysis Model and Data Exploration](https://yt-comment-sentiment-analysis.herokuapp.com/)
## Project Overview
This data tool is a use case demonstration of a text sentiment analysis classifier. The web app displays comments from a list of upcoming movie trailer videos off youtube and performs a sentiment analysis on them with a tensorflow built model with an evaluation of 84% accuracy. The app simultaneously allows for the use of another model, an SKLearn built model with an evaluation of 90% accuracy, to re-classify the pre-evaluated sentiments and compare performance.

### How to use the Web App:
* Paste in a comment from the data frame to compare the tensorflow model's performance against the sklearn model. Or simply paste in a test comment as if you were commenting under a trailer video. The SK model will predict its sentiment and display it.

## Data Collection
I used Selenium to scrape a list of youtube videos' relevant data. With each video, the following data was scraped:
* Title
* Views
* Video Likes
* Video Dislikes
* Top 50 Comments

I also utilized the IMDB reviews dataset for the creation of my classification model. It consists of 25k positive reviews and 25k negative reviews.

## Data Cleaning
Here are some tasks that were tackled in the cleaning process:
* Parsed title to clean text.
* Feature engineered comment length and language.
* (IMDB dataset)Parsed review to remove random html tags.

## Exploratory Data Analysis
![alt text](https://github.com/danteairdharris/CommentSentimentAnalysis/blob/master/cloud_still.png)
![alt text](https://github.com/danteairdharris/CommentSentimentAnalysis/blob/master/bar.png)

## Model
The tensorflow model utilizes a Sequential architecture with 2 each Dense and Dropout hidden layers. SKLearn OneHotEncoder and Google Universal Sentence Encoder are used to encode and embed text respectively. The final model is evaluated at 0.84 accuracy with 0.36 validation loss. The SKLearn model performs better at an evaluated 0.90 accuracy mostly due to its use of Term Frequency x Inverse Document Frequency vectorization which helps to reduce the prevalence of frequent unimportant terms like 'the' and 'movie' in our bag of words representation. The comment sentiments were pre evaluated using the Tensorflow model but the web app gives the user the opportunity to re-classify them (or any text) using the SKLearn model to compare performance.

## Productionization
Used Streamlit to quickly and cleanly deploy the model for use.

