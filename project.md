---
title: "Human Activity Predictive Modelling"
author: "Stefan Putra Lionar"
date: "12 March 2018"
output:
  html_document:
    keep_md: true
---



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

## Data Preprocessing

First, import our training and test set.


```r
train.raw <- read.csv("pml-training.csv")
test.raw <- read.csv("pml-testing.csv")
```

We exclude variables with more than NA and blank values more than 10 %.


```r
maxNAPerc = 10
maxNACount <- nrow(train.raw) / 100 * maxNAPerc
col_subset<-which(colSums(is.na(train.raw) | train.raw=="")>maxNACount)
train <- train.raw[,-col_subset]
test <- test.raw[,-col_subset]
```

Let's check if there is any NA or blank values.

```r
sum(is.na(train) | train == "")
```

```
## [1] 0
```

```r
sum(is.na(test) | test == "")
```

```
## [1] 0
```

It seems there is no more NA or blank values. That is great! It means we don't have to do any imputation.

## Feature Engineering

In this project, we focus on building scalable model. First we exclude variables `X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp` due to the following reasons:

1. Those variables are not good indicators of the response variable
2. Those variables are series, hence it may worsen our model by introducing bias. Our model should focus to work on independent sample.


```r
train <- subset(train, select = -c(X,raw_timestamp_part_1,raw_timestamp_part_2, cvtd_timestamp))
test <- subset(test, select = -c(X,raw_timestamp_part_1,raw_timestamp_part_2, cvtd_timestamp))
```


We exclude `user_name` as new test data can have any user name. This left us with 1 factor variable `new_window` and the rest of numerical/integer variables. The numerical/integer variables can be shrinked down by performing PCA. Let's see if variable `new_window` has significant effect to response variable `classe` by performing chi-squared test.


```r
contingency_table <- table(train$new_window, train$classe)
chisq.test(contingency_table)
```

```
## 
## 	Pearson's Chi-squared test
## 
## data:  contingency_table
## X-squared = 0.7341, df = 4, p-value = 0.9471
```

With p-value as high as 0.94, we exclude variable `new_window`.

Let's perform PCA by firstly select numerical and integer columns on our training and test set.



```r
train_num_features <- train[,sapply(train,is.numeric) | sapply(train,is.integer)]

test_num_features <- subset(test, select = -c(problem_id))
test_num_features <- test_num_features[,sapply(test_num_features,is.numeric) | sapply(test_num_features,is.integer)]
```

Then perform PCA with centering and scaling.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
preProc <- preProcess(train_num_features, method = c("center", "scale","pca"), thres=0.95)
preProc
```

```
## Created from 19622 samples and 53 variables
## 
## Pre-processing:
##   - centered (53)
##   - ignored (0)
##   - principal component signal extraction (53)
##   - scaled (53)
## 
## PCA needed 26 components to capture 95 percent of the variance
```

We have shrinked our numerical/integer explanatory variables from 53 to 26 components, which catch 95% of the variance. Let's get these features and store them in new data frame `train_pca` and `test_pca`.


```r
train_pca_features <- predict(preProc, train_num_features)
test_pca_features <- predict(preProc, test_num_features)

train_pca <- cbind(train_pca_features, train$classe)
names(train_pca)[length(train_pca)] <- "classe"
test_pca <- cbind(test_pca_features)
```

## Training

We will train our classifier using RandomForest algorithm with 10-fold cross validation. We start with ntree = 200. The more ntree may pose our model to overfitting and will lengthen training time.


```r
library(doSNOW) #doSNOW library for parallel processing. This is to speed up training process.
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```
## Loading required package: snow
```

```r
cl <- makeCluster(4, type = "SOCK") #Set 4 parallels based on number of processor cores
registerDoSNOW(cl)

set.seed(1010) #set seed for reproducibility
train.control <- trainControl(method = "cv", #Set our train control
                               number = 10,
                               search = "grid")

rf <- train(classe ~., #Train our model
            data = train_pca,
            method = "rf",
            metrics = "Accuracy",
            trControl = train.control,
            #tuneGrid = tunegrid,
            ntree = 200)
rf
```

```
## Random Forest 
## 
## 19622 samples
##    26 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17659, 17661, 17659, 17659, 17660, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9833350  0.9789187
##   14    0.9813478  0.9764050
##   26    0.9775258  0.9715727
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
stopCluster(cl) #close session
```

We get our predictive model with cross validation accuracy 98.33%. This number is considered very high. Let's test it in the test set.

## Prediction


```r
classe.predict<- predict(rf, test_pca)

prediction <- data.frame()
prediction <- cbind.data.frame(test$problem_id,classe.predict)
prediction
```

```
##    test$problem_id classe.predict
## 1                1              B
## 2                2              A
## 3                3              B
## 4                4              A
## 5                5              A
## 6                6              E
## 7                7              D
## 8                8              B
## 9                9              A
## 10              10              A
## 11              11              B
## 12              12              C
## 13              13              B
## 14              14              A
## 15              15              E
## 16              16              E
## 17              17              A
## 18              18              B
## 19              19              B
## 20              20              B
```

