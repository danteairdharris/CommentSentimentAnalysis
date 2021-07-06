# [Youtube Comments Text Sentiment Analysis Model and Data Exploration](https://yt-comment-sentiment-analysis.herokuapp.com/)
## Project Overview
This data tool is a use case demonstration of a text sentiment analysis classifier. The web app displays comments from a list of upcoming movie trailer youtube videos and performs a sentiment analysis on them with model evaluation of 90% accuracy. 

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
![alt text](https://github.com/danteairdharris/BoxOfficeDS/blob/master/totalbudget.png)
![alt text](https://github.com/danteairdharris/BoxOfficeDS/blob/master/releasemonth.png)
![alt text](https://github.com/danteairdharris/BoxOfficeDS/blob/master/heatmap.png)

## Model
The model utilizes python library SciKit Learn to build a Random Forest Regression and Linear Regression trained on the aforementioned dataset.
* The features passed to the model include Release Month, Budget, Box Office Opening, and Rating. 
* The dataset is split into train and test sets with 20% of the data reserved for testing and 80% for training. 
* A grid search was used to find the optimal parameter set for the Random Forest Regression.
* The model determined the RF regression performed better with a MAE of ~ $45 Million

## Productionization
Used Streamlit to quickly and cleanly deploy the model for use.

