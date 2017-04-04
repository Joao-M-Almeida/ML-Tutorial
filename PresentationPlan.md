## What's this presentation going to be about? (1 min)

Hello everyone,

### Who am I

I'm João Almeida, I'm Software Engineer at CERN and I'm by no means an expert in Machine Learning.
I took a few courses in college, attended some talks, read a lot, did a few projects ...s

### This presentation's format:

There won't be any slides, I am going to show you a lot of code, some math and some plots.

We won't focus on the theoretical details behind the ML techiniques, but on their intuition and on how to apply them. We will do this with the help of Python and Scikit Learn, a very nice python ML library.


 <!-- but i’ve got rid of as much maths as possible and put in lots of pictures.

you only need to get a broad idea of what’s going on to be able to use it effectively.  -->

------

## What is Machine Learning? (1 min)

  ML is a set of techniques used to teach computers to learn from data.

  Computers learn from examples and experience instead of following hard coded rules.

The learning task can be varied:

- it might be grouping examples together
- Predicting some continuous valued of a new example (Regression): a typical example is predicting the price of a house based on its characteristics;
- Finding which examples have unexpected characteristics. One example is fraud detection in online purchases
- Taking new examples and assigning labels to them, for instance taking a picture of a fruit and saying whether it is an apple or a banana.
- Predicting the next value in a time series, for instance predicting stock prices.


## Supervised vs Unsupervised Learning (1min)

There are two distinct groups of ML applications, supervised learning and Unsupervised
The distinction is whether you have data to teach what you are learning.

### Supervised:
For instance when doing classification or regression you have data on what is supposed to be the output of your model for each example.

### Unsupervised Learning
Clustering is the typical example, you have a bunch of data and want to try to extract some knowledge but you don't know exactly what.
Clustering techniques allow you to find clusters "groups" of examples that are somehow similar,
<!--
#### Classification vs Regression

__Classifier__: Prefictor with a categorical

__Regressor__: Predictor with a continuous output

Imagine you a program to help you trade stocks and make money from it. There are two possible ways of facing this problem, you can make a problem to predict the price of the stock each day or you can make a program that just tells you whether you shoul __buy__, __sell__ or __do nothing__ each day.

## Unsupervised Learning

Unsupervised Learning is to try to learn something from data, but without having any labels -->

## The data (20 s)

A dataset is a set of examples used to train a Machine Learning model

An example contains information about an object or event;

The example is represented by its features.

I think this is more understandable with examples.

------

## Let's look at some code

### The Jupyter Notebook (1 min)

This is an environment that allows you to run some python code in blocks and see the results right away.

Very similar to the Matlab cells if you ever worked with them

... doing some imports necessary for later


------


## Boston House Prices

A very well known regression dataset, where we have a bunch of house features and want to predict the price of that house.

It's available inside scikit learn.

__Load dataset__

Here's some info about the dataset;

__Show dataframe:__

Here I'm using Pandas, another very cool python library to take a look at the dataset.

We can see a few examples with their features and the price of each house.


One very important step when doing machine learning is to understand the data and how each relates to each other.

To try to understand these relations we can plot the features against the price.
__Let's see some plots__

These plots show clearly that there are features which have a much higher correlation with the price of the house, for instance  average number of rooms vs the per capita crime.

However all plotted features seem to be somewhat correlated with the price.

------

## Using Linear Regression:

## Visualizing the resulting model:

We are in an higher dimention, we can't easily visualize this model, so we have to rely on metrics to estimate the model's performance.

We can also look at some examples and see how the model is performing.

Now let's look at a classification task before delving into how linear regression works.

## The Iris dataset (1 min)

This is a very common classification dataset, it's small and so is available inside scikit learn.

The Iris are a family of flowers and in this dataset we have examples from 3 different species.

__LOAD dataset__
These are the features and the 3 class labels;

__Look at dataframe__ (1min)

Here we can see the examples, the features and the labels.

_Is everyone understanding?_


__Plot data__ (20 s)

We can see here all the examples colored by class.

To start slowly we will focus only on the Iris Setosa and try only to classify each example as belonging to that species or not.

__New plot__ (40 s)

Let's look at the plot again, now with only two classes.

Notes:
Looking at this plot, if I have a new flower that would be at (4.5, 4.0)  what kind of Iris would you predict it is?
and here (7.0, 3.5)?
and here (7.0, 4.0)?

That's exactly what a machine learning algorithm does it uses the available data to make predictions, some times it get's it right other times it fails.

------

## Logistic Regression:

As you know naming things is one of the hardest problems in computer science and someone chose an awful name for this algorithm, because the Logistic Regression doesn't do Regression but classification.

__Run the algorithm__

As you the model follows exactly the same API and this is true for all scikit learn models which makes it very easy to use and to change models we are working on.

Here the accuracy is the % of examples we classified correctly.

We can take a look at what the model is doing:

__plot data with decision boundary__

As you can see it draws a linear decision boundary where all examples in one side are classified to one class and on the other side to the other.

--------

## Linear Regression and Logistic Regression

The theory:

### Linear Regression


#### How to find the Weights


### Logistic Regression


#### Sigmoid Function

__plot sigmoid__


----------

Now that we have covered the basics of machine learning let's play with a real world dataset.
## Let's play with a real world dataset

## Look at Data

## Look at the feature Histograms

## look at how they are related to each other

## Overfitting

## KNN

## Curse of Dimensionality

## PCA

## Predicting Tobacco
