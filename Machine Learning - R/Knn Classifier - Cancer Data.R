setwd("C:/Users/pierr/Desktop/R_MachineLearning")

download.file("https://github.com/stedy/Machine-Learning-with-R-datasets/raw/master/wisc_bc_data.csv",
              destfile = "Cancer_data.csv")

Cancer <- read.csv("Cancer_data.csv", stringsAsFactors = FALSE, header = TRUE)
head(Cancer)
str(Cancer)

wbcd <- Cancer[-1]

table(wbcd$diagnosis)

wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))

round(prop.table(table(wbcd$diagnosis)) * 100, digits = 2)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

library(dplyr)

wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
wbcd_prep <- bind_cols(wbcd[1], wbcd_n)

head(wbcd_prep)

library(caret)

set.seed(123456)

ind <- createDataPartition(wbcd_prep$diagnosis, p = 0.7, list =FALSE)
train <- wbcd_prep[ind, ]
test <- wbcd_prep[-ind, ]

library(class)

wbcd_test_pred <- knn(train[-1], test[-1], cl = train[,1], k = 23)

library(gmodels)

CrossTable(x = test[, 1], y = wbcd_test_pred, prop.chisq = FALSE)
confusionMatrix(test[, 1], wbcd_test_pred)


wbcd_z <- as.data.frame(lapply(wbcd[2:31], scale))
wbcd_pz <- bind_cols(wbcd[1], wbcd_z)

ind <- createDataPartition(wbcd_pz$diagnosis, p = 0.7, list =FALSE)
train_z <- wbcd_pz[ind, ]
test_z <- wbcd_pz[-ind, ]

wbdc_test_pred_z <- knn(train_z[-1], test_z[-1], cl = train_z[, 1], k = 23)

confusionMatrix(test_z[, 1], wbdc_test_pred_z)
