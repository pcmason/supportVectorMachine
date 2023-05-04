#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:43:15 2023

Support Vector Machine (SVM) linear classifier using optimization routines from SciPy

@author: paulmason
"""

import numpy as np
#Optimization imports
from scipy.optimize import Bounds, BFGS
from scipy.optimize import LinearConstraint, minimize
#Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
#Generating datasets import
import sklearn.datasets as dt

#Constant that is close to 0, but not exactly 0
ZERO = 1e-7

#Function to plot all 9 pairs separated into 2 groups: +1 and -1
def plot_x(x, t, alpha= [], C = 0):
    sns.scatterplot(dat[:, 0], dat[:, 1], style = labels,
                    hue = labels, markers = ['s', 'P'],
                    palette = ['magenta', 'green'])
    if len(alpha) > 0:
        alpha_str = np.char.mod('%.1f', np.round(alpha, 1))
        ind_sv = np.where(alpha > ZERO)[0]
        for i in ind_sv:
            plt.gca().text(dat[i, 0], dat[i, 1]-.25, alpha_str[i])
            
#Objective function, LaGrange dual in this case
def lagrange_dual(alpha, x, t):
    result = 0
    ind_sv = np.where(alpha > ZERO)[0]
    for i in ind_sv:
        for k in ind_sv:
            result = result + alpha[i] * alpha[k] * t[i] * t[k] * np.dot(x[i, :], x[k, :])
    result = 0.5 * result - sum(alpha)
    return result   

#Find the optimal values of alpha
def optimize_alpha(x, t, C):
    m, n = x.shape
    np.random.seed(1)
    #Initialize alphas to random values
    alpha_0 = np.random.rand(m) * C
    #Define the constraint
    linear_constraint = LinearConstraint(t, [0], [0])
    #Define the bounds
    bounds_alpha = Bounds(np.zeros(m), np.full(m, C))
    #Find the optimal value of alpha
    result = minimize(lagrange_dual, alpha_0, args = (x, t), method = 'trust-constr',
                      hess = BFGS(), constraints = [linear_constraint], 
                      bounds = bounds_alpha)
    #The optimized value of alpha lies in result.x
    alpha = result.x
    return alpha


#Both w and w0 are necessary for the creation of the hyperplane
#Method to find the weight vector
def get_w(alpha, t, x):
    m = len(x)
    #Get all support vectors
    w = np.zeros(x.shape[1])
    for i in range(m):
        w = w + alpha[i] * t[i] * x[i, :]
    return w

#Method to compute constant w0
def get_w0(alpha, t, x, w, C):
    C_numeric = C - ZERO
    #Indices of support vectors with alpha < C
    ind_sv = np.where((alpha > ZERO) & (alpha < C_numeric))[0]
    w0 = 0.0
    for s in ind_sv:
        w0 = w0 + t[s] - np.dot(x[s, :], w)
    #Take the average
    w0 = w0 / len(ind_sv)
    return w0

#Take an array of test points with w & w0 and classify various points
def classify_points(x_test, w, w0):
    #get y(x_test)
    predicted_labels = np.sum(x_test * w, axis = 1) + w0
    predicted_labels = np.sign(predicted_labels)
    #Assign a lable arbitrarily a +1 if it is zero
    predicted_labels[predicted_labels == 0] = 1
    return predicted_labels

#Calculate the misclassification rate
def misclassification_rate(labels, predictions):
    total = len(labels)
    errors = sum(labels != predictions)
    return errors / total * 100

#Function to plot hyperplane
def plot_hyperplane(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    y_coord = -w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, y_coord, color = 'red')

#Function to plot the soft margin
def plot_margin(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    ypos_coord = 1/w[1] - w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, ypos_coord, '--', color = 'green')
    yneg_coord = -1/w[1] - w0/w[1] - w[0]/w[1] * x_coord 
    plt.plot(x_coord, yneg_coord, '--', color = 'magenta')     
    
#Function to run the SVM
def display_SVM_result(x, t, C):
    #Get the alphas
    alpha = optimize_alpha(x, t, C)
    #Get the weights
    w = get_w(alpha, t, x)
    w0 = get_w0(alpha, t, x, w, C)
    plot_x(x, t, alpha, C)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plot_hyperplane(w, w0)
    plot_margin(w, w0)
    plt.xlim(xlim)
    plt.ylim(ylim)
    #Get the misclassification error and display it as title
    predictions = classify_points(x, w, w0)
    err = misclassification_rate(t, predictions)
    title = 'C = ' + str(C) + ', Errors: ' + '{:.1f}'.format(err) + '%'
    title = title + ', total SV = ' + str(len(alpha[alpha > ZERO]))
    plt.title(title)
    
 

#Create random set of data points and assocaited labels for each pair
dat = np.array([[0, 3], [-1, 0], [1, 2], [2, 1], [3, 3], [0, 0], [-1, -1], [-3, 1], [3, 1]])
labels = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1])    
plot_x(dat, labels)
plt.show()
display_SVM_result(dat, labels, 100)
plt.show()


#Create new dataset and labels from scipy
dat, labels = dt.make_blobs(n_samples = [20, 20],
                            cluster_std = 1,
                            random_state = 0)
labels[labels == 0] = -1
plot_x(dat, labels)

#Define different values of C and run the code
fig = plt.figure(figsize = (8, 25))

i = 0
C_array = [1e-2, 100, 1e5]

for C in C_array:
    fig.add_subplot(311+i)
    display_SVM_result(dat, labels, C)
    i = i + 1


