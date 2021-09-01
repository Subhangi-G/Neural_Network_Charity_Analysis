# Neural_Network_Charity_Analysis

## Overview of Project 

### Purpose

The purpose of this project is to create a binary classifier that will predict whether an applicant for Alphabet Soup Charity, when funded, will successfully utilize its funds.\
This was achieved by creating a neural network model which was trained and tested on a dataset of more than 34,000 organizations that had already received funding from Alphbet Soup, and their funds utilization is known.

### Resources used
- Data : Resources\Charity_Data.csv
- Software : scikit-learn 0.24.1, Numpy 1.20.1, Pandas 1.2.4, hvplot 0.7.3, Plotly 5.1.0, Tensorflow 2.6.0

## Results and Analysis
A csv file containing metadata of over 34,000 organizations, that received funds from Alphabet Soup, was used as the data file. The data was loaded into Pandas a DataFrame. The features included:
- EIN and NAME - Identification columns
- APPLICATION_TYPE — Alphabet Soup application type
- AFFILIATION — Affiliated sector of industry
- CLASSIFICATION — Government organization classification
- USE_CASE — Use case for funding
- ORGANIZATION — Organization type
- STATUS — Active status
- INCOME_AMT — Income classification
- SPECIAL_CONSIDERATIONS — Special consideration for application
- ASK_AMT — Funding amount requested
- IS_SUCCESSFUL — Was the money used effectively

The main steps in the project are described below.

### Data Preprocessing
Answers to the following questions regarding the DataFrame shown below, will assist in comprehending the process.

(pic. of the original dataframe).

- _**What variable(s) are considered the target(s) for your model?**_

The question asked in this project addresses whether the applicant will be successful upon recieving the funds, therefore the variable "IS_SUCCESSFUL" was set as the target.

- _**What variable(s) are considered to be the features for your model?**_

Every variable other than "IS_SUCCESSFUL", and the identification columns EIN and name were considered to be the features of the neural network model.

- _**What variable(s) are neither targets nor features, and should be removed from the input data?**_

The identification columns are expected to not play a part on determining the success of an organization, and therefore were removed.

- _**Further pre-processing steps**_

There were no null values in the dataset. The categorical variables "APPLICATION_TYPE" and "CLASSIFICATION"  had a large number of unique values, and therefore their numbers were reduced by bucketing the comparitively rare values into a single "other" category.\
The object variables were converted to numeric using one-hot encoder. This resulted in a total of 44 input features .

The dataset was divided into the training and testign set, and standardized using standard scaler.

The neural network model was then defined, complied, trained on the scaled train set, and evaluated on the scaled test set.

### Compiling, Training, and Evaluating the Model

- _**How many neurons, layers, and activation functions did you select for your neural network model, and why?**_

There were three hidden layers. As there were 44 input parameters, the first hidden layer had approximately three times the number of neurons (130). The second hidden layer had half the neurons of the first layer (65), and the last hidden layer had 10 neurons.\
The popular non-linear ReLU activation function was used for all the hidden layers.\
The output layer had a sigmoid activation function because the expected output of the algorithm is binary.

(pic of nn1)

- _**Were you able to achieve the target model performance?**_

The model was trained on the scaled train set for 50 epochs, and reached a performance accuracy of 72.6% when evaluated on the scaled test-data (fig. below). This was short of the accuracy goal of 75%.\
The results from neural network model, nn_1, was stored as a HDF5 file named AlphabetSoupCharity_Optimization.h5. 

- _**What steps did you take to try and increase model performance?**_

The model was recreated two more times incorporating changes.\
The second attempt of the model, nn_2, was defined by just two hidden layers (Fig. below).

(pic of nn2).

The number of neurons in the first layer was double the number of input parameters. The second hidden layer had less than half the number of neurons of the first layer.\
The activation function used for the hidden layers was a version of ReLU called the parameterized ReLU. The sigmoid activation function was used for the output layer.\
The model was compiled and trained on the scaled train data over 100 epochs.
The model when evaluated on the scaled test data gave an accuracy of 72.5%, very similar to the previous model nn_1 (Fig. below).

(pic of eval nn2)

In the third attempt the model, nn_3, was defined again by two hidden layers.

(pic of nn3)

The number of neurons in the first layer was three times the number of input parameters. The second hidden layer had approximately half the number of neurons of first layer.\
The activation function used for the hidden layers was tanh. The sigmoid activation function was used for the output layer.\
The model was compiled and trained on the scaled train data over 100 epochs.
The model when evaluated on the scaled test data gave an accuracy of 72.1%, very similar to the previous models.

(pic of eval nn3)


## Summary

All the models used here gave approximately the same performance accuracy of around 72%. Changes incorportated in the different models included changing the number of hidden layers, the number of neurons at each layer, the activation function for the hidden layers, and the number of epochs used for training the model on the scaled train data.\
Comparing the accuracy acheived during training and evaluation confirms that model was not over fit. Pre-processing the data differently or using a different model may have improved the accuracy of prediction.

A good model to try solving the classification problem would be an ensemble method like the Random Forest Classifier because the majority of the variables in this data are categorical with discrete values. A number of weak learners, in this case decision trees formed from a sample of observations, will reach a conclusion on whether a given applicant will be sucessful or not based on a sample of the features.\
The final decision is based on voting, which combines the prediction of many decision trees. This is based on the philosophy that many mediocre models taken together will reach a better or more accurate conclusion compared to one very good model, like the deep learning model used in this project. 