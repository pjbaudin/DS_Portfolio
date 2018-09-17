getwd()
setwd("./GitHub/DS_Portfolio/Machine Learning - R/Naive_Bayes")
getwd()



df_train <- read.csv("train.csv")
df_test <- read.csv("test.csv")

df_train$label <- as.factor(df_train$label)

# Brief exploration
dim(df_train)
dim(df_test)
prop.table(table(df_train$label))

library(e1071) 


m <- naiveBayes(label ~ ., data = df_train, laplace = 0)

pred <- predict(m, df_test, type = "class")

pred_subm <- data.frame(ImageId = 1:nrow(df_test), label = pred)
head(pred_subm)

write.csv(pred_subm, "submission.csv", row.names = FALSE)

