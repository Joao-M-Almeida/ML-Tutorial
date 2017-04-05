""" Utils used during the tutorail talk.

TODO:
- Improve imports
- clean up linter warnings
"""

import warnings
from time import time

import numpy as np
#import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split


warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

boston_dataset = datasets.load_boston()



def predict_mesh(X, clf, gap=0.3, h=0.01):
    x_min, x_max = X[:, 0].min() - gap, X[:, 0].max() + gap
    y_min, y_max = X[:, 1].min() - gap, X[:, 1].max() + gap

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return (xx, yy, Z)


def describe_example_boston_dataset(example):
    """ Print description of an example of the boston data set """
    if example.shape != (13,):
        print("Unable to describe")
        return
    for i in range(0, 13):
        print("Feature: {:8s} - {:8.2f}".format(boston_dataset.feature_names[i], example[i]))

def plot_boston_dataset(X, Y):
    """ Look at the data """

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.scatter(x=X[:, 0], y=Y, alpha=0.3)
    ax1.set_xlabel('per capita crime', fontsize=14)
    ax1.set_ylabel('Median value', fontsize=14)


    ax2.scatter(x=X[:, 4], y=Y, alpha=0.3)
    ax2.set_xlabel('nitric oxides concentration', fontsize=14)
    ax2.set_ylabel('Median value', fontsize=14)

    ax3.scatter(x=X[:, 5], y=Y, alpha=0.3)
    ax3.set_xlabel('average number of rooms ', fontsize=14)
    ax3.set_ylabel('Median value', fontsize=14)


    ax4.scatter(x=X[:, 9], y=Y, alpha=0.3)
    ax4.set_xlabel('full-value property-tax per 10.000$', fontsize=14)
    ax4.set_ylabel('Median value', fontsize=14)


    ax5.scatter(x=X[:, 6], y=Y, alpha=0.3)
    ax5.set_xlabel('AGE', fontsize=14)
    ax5.set_ylabel('Median value', fontsize=14)


    ax6.scatter(x=X[:, 12], y=Y, alpha=0.3)
    ax6.set_xlabel('LSTAT', fontsize=14)
    ax6.set_ylabel('Median value', fontsize=14)

    plt.show()

def plot_whisky_histograms(whisky_dataframe):
    fig = plt.figure(figsize=(20,15))
    ax1 = fig.add_subplot(3,4,1)
    ax2 = fig.add_subplot(3,4,2)
    ax3 = fig.add_subplot(3,4,3)
    ax4 = fig.add_subplot(3,4,4)
    ax5 = fig.add_subplot(3,4,5)
    ax6 = fig.add_subplot(3,4,6)
    ax7 = fig.add_subplot(3,4,7)
    ax8 = fig.add_subplot(3,4,8)
    ax9 = fig.add_subplot(3,4,9)
    ax10 = fig.add_subplot(3,4,10)
    ax11 = fig.add_subplot(3,4,11)
    ax12 = fig.add_subplot(3,4,12)

    ax1.hist(x=whisky_dataframe['Smoky'], bins=range(0,5), rwidth=0.85, align='right')
    ax1.set_title("Smoky", fontsize=15)
    ax1.set_xticks([0,1,2,3,4,5])
    ax1.set_ylabel('Frequency')

    ax2.hist(x=whisky_dataframe['Honey'], bins=range(0,5), rwidth=0.85, align='right')
    ax2.set_title("Honey", fontsize=15)
    ax2.set_xticks([0,1,2,3,4])
    ax2.set_ylabel('Frequency')

    ax3.hist(x=whisky_dataframe['Body'], bins=range(0,5), rwidth=0.85, align='right')
    ax3.set_title("Body", fontsize=15)
    ax3.set_xticks([0,1,2,3,4])
    ax3.set_ylabel('Frequency')

    ax4.hist(x=whisky_dataframe['Nutty'], bins=range(0,5), rwidth=0.85, align='right')
    ax4.set_title("Nutty", fontsize=15)
    ax4.set_xticks([0,1,2,3,4])
    ax4.set_ylabel('Frequency')

    ax5.hist(x=whisky_dataframe['Malty'], bins=range(0,5), rwidth=0.85, align='right')
    ax5.set_title("Malty", fontsize=15)
    ax5.set_xticks([0,1,2,3,4])
    ax5.set_ylabel('Frequency')

    ax6.hist(x=whisky_dataframe['Fruity'], bins=range(0,5), rwidth=0.85, align='right')
    ax6.set_title("Smoky", fontsize=15)
    ax6.set_xticks([0,1,2,3,4])
    ax6.set_ylabel('Frequency')

    ax7.hist(x=whisky_dataframe['Sweetness'], bins=range(0,5), rwidth=0.85, align='right')
    ax7.set_title("Sweetness", fontsize=15)
    ax7.set_xticks([0,1,2,3,4])
    ax7.set_ylabel('Frequency')

    ax8.hist(x=whisky_dataframe['Medicinal'], bins=range(0,5), rwidth=0.85, align='right')
    ax8.set_title("Medicinal", fontsize=15)
    ax8.set_xticks([0,1,2,3,4])
    ax8.set_ylabel('Frequency')

    ax9.hist(x=whisky_dataframe['Tobacco'], bins=range(0,5), rwidth=0.85, align='right')
    ax9.set_title("Tobacco", fontsize=15)
    ax9.set_xticks([0,1,2,3,4])
    ax9.set_ylabel('Frequency')

    ax10.hist(x=whisky_dataframe['Spicy'], bins=range(0,5), rwidth=0.85, align='right')
    ax10.set_title("Spicy", fontsize=15)
    ax10.set_xticks([0,1,2,3,4])
    ax10.set_ylabel('Frequency')

    ax11.hist(x=whisky_dataframe['Winey'], bins=range(0,5), rwidth=0.85, align='right')
    ax11.set_title("Winey", fontsize=15)
    ax11.set_xticks([0,1,2,3,4])
    ax11.set_ylabel('Frequency')

    ax12.hist(x=whisky_dataframe['Floral'], bins=range(0,5), rwidth=0.85, align='right')
    ax12.set_title("Floral", fontsize=15)
    ax12.set_xticks([0,1,2,3,4])
    ax12.set_ylabel('Frequency')

    plt.show()

def plot_whiky_body_correlation(whisky_dataframe):
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(3,3,1)
    ax2 = fig.add_subplot(3,3,2)
    ax3 = fig.add_subplot(3,3,3)
    ax4 = fig.add_subplot(3,3,4)
    ax5 = fig.add_subplot(3,3,5)
    ax6 = fig.add_subplot(3,3,6)
    ax7 = fig.add_subplot(3,3,7)
    ax8 = fig.add_subplot(3,3,8)
    ax9 = fig.add_subplot(3,3,9)

    ax1.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Sweetness'], alpha=0.1, s=500)
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_xlabel('Body')
    ax1.set_ylabel('Sweetness')

    ax2.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Smoky'], alpha=0.1, s=500)
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_xlabel('Body')
    ax2.set_ylabel('Smoky')

    ax3.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Medicinal'], alpha=0.1, s=500)
    ax3.set_xticks([0, 1, 2, 3, 4])
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_xlabel('Body')
    ax3.set_ylabel('Medicinal')

    ax4.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Tobacco'], alpha=0.1, s=500)
    ax4.set_xticks([0, 1, 2, 3, 4])
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_xlabel('Body')
    ax4.set_ylabel('Tobacco')

    ax5.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Floral'], alpha=0.1, s=500)
    ax5.set_xticks([0, 1, 2, 3, 4])
    ax5.set_yticks([0, 1, 2, 3, 4])
    ax5.set_xlabel('Body')
    ax5.set_ylabel('Floral')

    ax6.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Fruity'], alpha=0.1, s=500)
    ax6.set_xticks([0, 1, 2, 3, 4])
    ax6.set_yticks([0, 1, 2, 3, 4])
    ax6.set_xlabel('Body')
    ax6.set_ylabel('Fruity')

    ax7.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Nutty'], alpha=0.1, s=500)
    ax7.set_xticks([0, 1, 2, 3, 4])
    ax7.set_yticks([0, 1, 2, 3, 4])
    ax7.set_xlabel('Body')
    ax7.set_ylabel('Nutty')

    ax8.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Malty'], alpha=0.1, s=500)
    ax8.set_xticks([0, 1, 2, 3, 4])
    ax8.set_yticks([0, 1, 2, 3, 4])
    ax8.set_xlabel('Body')
    ax8.set_ylabel('Malty')

    ax9.scatter(x = whisky_dataframe['Body'], y=whisky_dataframe['Winey'], alpha=0.1, s=500)
    ax9.set_xticks([0, 1, 2, 3, 4])
    ax9.set_yticks([0, 1, 2, 3, 4])
    ax9.set_xlabel('Body')
    ax9.set_ylabel('Winey')

    plt.show()


print("Import done")
