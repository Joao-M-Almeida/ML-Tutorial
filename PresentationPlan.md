## What's this presentation going to be about? (1 min)

Hello everyone,

### Who am I

I'm João Almeida, I'm Software Engineer at CERN and I'm by no means an expert in Machine Learning.
I took a few courses in college, attended some talks, read a lot, did a few projects ...s

### This presentation's format:

There won't be any slides, I am going to show you a lot of code, some math and some plots.

We won't focus on the theoretical details behind the ML techniques, but on their intuition and on how to apply them. We will do this with the help of Python and Scikit Learn, a very nice python ML library.


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

-------
_4 min_

## Supervised vs Unsupervised Learning (1min)

There are two distinct groups of ML applications, supervised learning and Unsupervised
The distinction is whether you have data to teach what you are learning.




### Supervised:
For instance when doing classification or regression you have data on what is supposed to be the output of your model for each example.


<!-- __Classification vs Regression__
Imagine you a program to help you trade stocks and make money from it. There are two possible ways of facing this problem, you can make a problem to predict the price of the stock each day or you can make a program that just tells you whether you shoul __buy__, __sell__ or __do nothing__ each day. -->

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

------
 4 min
-------
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

## The data (20 s)

A dataset is a set of examples used to train a Machine Learning model

An example contains information about an object or event;

The example is represented by its features.

I think this is more understandable with examples.
## Let's see a real example:

Show dataframe and explain what are features, examples and labels

------

One very important step when doing machine learning is to understand the data and how each relates to each other.

To try to understand these relations we can plot the features against the price.
__Let's see some plots__

These plots show clearly that there are features which have a much higher correlation with the price of the house, for instance  average number of rooms vs the per capita crime.

However all plotted features seem to be somewhat correlated with the price.

------

__Until here 10 min__

----------

## Using Linear Regression:

Who has heard of it? who has used it? maybe with Excel?


## Visualizing the resulting model:

We are in an higher dimention, we can't easily visualize this model, so we have to rely on metrics to estimate the model's performance.

We can also look at some examples and see how the model is performing.

Now let's look at a classification task before delving into how linear regression works.

------

__Until here ~ 13-15 min__

----------
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

It's called logistic regression but it is used for classification, there is a reason behind this, but it's a long story. In short naming things in Computer Science is hard


__Run the algorithm__

As you the model follows exactly the same API and this is true for all scikit learn models which makes it very easy to use and to change models we are working on.

Here the accuracy is the % of examples we classified correctly.

We can take a look at what the model is doing:

__plot data with decision boundary__

As you can see it draws a linear decision boundary where all examples in one side are classified to one class and on the other side to the other.

--------

## Linear Regression and Logistic Regression

### Linear Regression

The output is a Linear combination of the features

Each feature has a weight, which can be positive or negative and the sum of all of the product between weights and features is the output of this model.

There the w_0 is the offset and we create a 'feature' x_0 with a value 1 we get a vectorized version.

#### How to find the Weights

Now that we understand how the models makes the predictions, how do we fit the model to the data? how do the find the weights that minimize the error?

Usually we use a method called Ordinary least squares where essentially we minimize the sum of the squared error for each datapoint.

It's an optimization problem we want to find the weights that minimize the error over all the examples we have for training.


### Logistic Regression

We want to use the same linear model  but now build a classifier.

We want this classifier to go from a continuous value, the output of the linear regression, and go to a label.
Today we will focus only in binary labels.

For that we use the Logistic/Sigmoid function .

It has some interesting properties:
- monotonous
- continuous
- limited between 0 and 1


#### Sigmoid Function

__plot sigmoid__


----------

Now that we have covered the basics of machine learning let's play with a real world dataset.
## Let's play with a real world dataset
Let's take the knowledge we gained and try to apply it to a real world dataset.

## Look at Data

## Look at the feature Histograms

## look at how they are related to each other

__First Look at Scotch data: 5 min__

--------
TODO how to connect this?s

# Small detour


## Curse of Dimensionality

### More features != Better data

For instance let's imagine your trying to classify different types of fruit, do you think having more features would improve our model?
For instance the name of the person that picked the fruit? Or his age? or whether they are vegetarian?
In theory if you added these features the model should just ignore them. However to understand if it

__Curse of Dimensionality: 3 min__

--------


## Feature Selection and Extraction

### PCA
 eigenvectors of covariance matrix

Principal components are the directions of largest variance

The eigenvectors with the largest eigenvalues are the principal components


__Model Complexity: 3 min__

------
### Model Complexity



__Model Complexity: 2 min__

--------
## Back to Scotch

### PCA to Scotch


__PCA to Scotch: 1 min__

--------
## Predicting Tobacco


#### Confusion Matrices

__Predicting Tobacco: 5 min__

--------
###  Cross validation

__Cross validation: 2 min__

--------
## Finishing remarks

I've been lying to you, I've been hiding most of the problems you might face when working with machine learning. However the goal of this talk was to make you interested in it not to scare you away.


__THE END__
-------
## Overfitting

## KNN
it does not attempt to construct a general internal model, but simply stores instances of the training data.
Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

 3 min
-------
