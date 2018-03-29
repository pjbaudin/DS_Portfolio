# Set workign directory
setwd("C:/Users/pierr/Documents/Github/Kaggle/Titanic")

# Load library
library(dplyr)
library(stringr)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Load titanic data
train <- read.csv("./Dataset/train.csv")
test <- read.csv("./Dataset/test.csv")

head(train)
str(train)

head(test)
str(test)

# Creating features
test$Survived <- NA
combi <- bind_rows(train, test)

combi$Name <- as.character(combi$Name)

combi$Title <- sapply(combi$Name,
                      FUN = function(x) {str_trim(strsplit(x, split='[,.]')[[1]][2], side = "both")})

combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

combi$Title <- as.factor(combi$Title)

combi$FamilySize <- combi$SibSp + combi$Parch + 1

combi$Surname <- sapply(combi$Name,
                        FUN = function(x) {str_trim(strsplit(x, split='[,.]')[[1]][1], side = "both")})

# Predicting missing Age with rpart
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], 
                method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

combi$Embarked[which(combi$Embarked == '')] = "S"
combi$Embarked <- as.factor(combi$Embarked)

combi$Fare[which(is.na(combi$Fare))] <- median(combi$Fare, na.rm=TRUE)

train <- combi[1:891,]
test <- combi[892:1309,]

# Fit model
library(randomForest)

# Using rpart
fit_rpart <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize,
             data = train, method = "class")

fancyRpartPlot(fit_rpart)

pred_rpart <- predict(fit_rpart, test, type = "class")

submit <- data.frame(PassengerID = test$PassengerId, Survived = pred_rpart)
write.csv(submit, file = "submission_v01_rpart.csv", row.names = FALSE)

# Using random Forest
set.seed(817)

fit_RF <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                       data = train,
                       importance = TRUE,
                       ntree = 2000)
varImpPlot(fit_RF)

pred_RF <- predict(fit_RF, test)

submit <- data.frame(PassengerID = test$PassengerId, Survived = pred_RF)
write.csv(submit, file = "submission_v01_RF.csv", row.names = FALSE)

# Using forest of conditional inference tree
library(party)

set.seed(817)

fit_cF <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                  data = train,
                  controls = cforest_unbiased(ntree = 2000, mtry = 3))

pred_cF <- predict(fit_cF, test, OOB = TRUE, type = "response")

submit <- data.frame(PassengerID = test$PassengerId, Survived = pred_cF)
write.csv(submit, file = "submission_v01_cF.csv", row.names = FALSE)