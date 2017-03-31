## What's this presentation going to be about?

Hello everyone,

### Who am I

I'm João almeida, I'm Software Engineer at CERN and I'm by no means an expert in Machine Learning.
I took a few courses in college, attended some talks, read a lot, did a few projects ...

### This presentation's format:

There won't be any slides, I am going to show you a lot of code, some math and some plots.

We won't focus on the theoretical details behind the ML techiniques, but on their intuition and on how to apply them. We will do this with the help of Python and Scikit Learn, a very nice python ML library.


but i’ve got rid of as much maths as possible and put in lots of pictures.

you only need to get a broad idea of what’s going on to be able to use it effectively.

## What is Machine Learning?

  ML is a set of techniques used to teach computers to learn from data.

  Computers learn from examples and experience instead of following hard coded rules.

The learning task can be varied:

- it might be grouping examples together
- Predicting some continuous valued of a new example (Regression): a typical example is predicting the price of a house based on its characteristics;
- Finding which examples have unexpected characteristics. One example is fraud detection in online purchases
- Taking new examples and assigning labels to them, for instance taking a picture of a fruit and saying whether it is an apple or a banana.
- Predicting the next value in a time series, for instance predicting stock prices.

### Datasets



## Supervised vs Unsupervised Learning

There are two distinct groups of ML applications, supervised learning and Unsupervised
The distinction is whether you have data to teach what you are learning.

### Supervised:
For instance when doing classification or regression you have data on what is supposed to be the output of your model for each example.

### Unsupervised Learning


#### Classification vs Regression

__Classifier__: Prefictor with a categorical

__Regressor__: Predictor with a continuous output

Imagine you a program to help you trade stocks and make money from it. There are two possible ways of facing this problem, you can make a problem to predict the price of the stock each day or you can make a program that just tells you whether you shoul __buy__, __sell__ or __do nothing__ each day.

## Unsupervised Learning

Unsupervised Learning is to try to learn something from data, but without having any labels
