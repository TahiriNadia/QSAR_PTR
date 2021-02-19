setwd('../../data/after_feature_selection')

library(MASS)

view_data <- function(data){
  print(head(data))
  print(ncol(data))
  print(nrow(data))
}

# Read dataset
qsar_train <- read.csv('trainset_105_after_feature_selection.csv', header=TRUE, sep=";")
qsar_test <- read.csv('testset_105_after_feature_selection.csv', header=TRUE, sep=";")

view_data(qsar_train)

# TRAIN
data_train <- qsar_train[,2:ncol(qsar_train)]
x_train <- qsar_train[,3:ncol(qsar_train)]
y_train <- qsar_train$Ratio.Ln

# TEST
data_test <- qsar_test[,2:ncol(qsar_test)]
x_test <- qsar_test[,3:ncol(qsar_test)]
y_test <- qsar_test$Ratio.Ln


# ensure results are repeatable
set.seed(7)
# load the library
install.packages("mlbench")
library(mlbench)
library(caret)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Ratio.Ln~., data=data_train, preProcess="scale", trControl=control)



library(SuperLearner)
listWrappers()

# Set the seed
set.seed(150)

rsq <- function (x, y) cor(x, y) ^ 2

# TRAIN 
# Fit the ensemble model
model <- SuperLearner(y_train,
                      x_train,
                      family=gaussian(),
                      SL.library=list("SL.randomForest",
                                      "SL.ranger",
                                      "SL.glmnet",
                                      "SL.bartMachine",
                                      "SL.ksvm",
                                      "SL.nnet"))

# Return the model
model
rsq(model$SL.predict, y_train)
RMSE(model$SL.predict, y_train)


# CROSS-VALIDATION (k = 5)
cv5.model <- CV.SuperLearner(y_train,
                            x_train,
                            V=5,
                            SL.library=list("SL.randomForest",
                                            "SL.ranger",
                                            "SL.glmnet",
                                            "SL.bartMachine",
                                            "SL.ksvm",
                                            "SL.nnet"))

rsq(cv5.model$SL.predict,cv5.model$Y)
RMSE(cv5.model$SL.predict,cv5.model$Y)


# TEST
preds <- predict(model,x_test)

rsq(preds$pred,y_test)
RMSE(preds$pred,y_test)

