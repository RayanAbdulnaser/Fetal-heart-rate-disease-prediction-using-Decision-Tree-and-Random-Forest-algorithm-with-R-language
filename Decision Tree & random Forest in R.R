#Libraries 
library(ggplot2)
library(rlang)
library(randomForest)
library(dplyr)
library(psych)
library(tidyverse)
library(GGally)
library(caret)
library(Amelia)
library(tidyr)
#Read Data
data <- read.csv("C:\\Users\\erere\\OneDrive\\Desktop\\ml\\Decision Tree & random Forest in R\\Cardiotocographic.csv")
str(data)
# ------------------- Data cleaning ------------------- 
# Remove missing values
data <- na.omit(data)
# Remove duplicates
data <- unique(data)
#showing the data after cleaning
data
#change NSP and Tendency in integer to factor format
data$NSPF <- factor(data$NSP)
#data$Tendencyf <- factor(data$Tendency)
xtabs(~NSPF,data=data)
# Count output elements number
table(data$NSPF)

#table(data$Tendencyf)
# Identify the numerical features
num_features <- sapply(data, is.numeric)

# Normalize the numerical features
data[, num_features] <- scale(data[, num_features])

#to see IV in scaled value or ggplot
summary(data[,2:5])
summary(data[,1:4])

# To apply the summary method and calculate the statistics for each variable
summary(data[, sapply(data, is.numeric)])

# Print the minimum and maximum values bounds that each variable can take.
sapply(data[, sapply(data, is.numeric)], function(x) c(min(x), max(x)))
#----------------------perform correlation-based featurs selection---------
# Select the features 
data <- arrange(data, LB, by_group = TRUE)
data <- select(data, LB, DS, ASTV, MSTV, ALTV, Width, Min, Max, Mode, Median, Tendency, NSPF)
colnames(data);

# ------------------- Data visualizations ------------------- 
pairs.panels(data[-1])

pairs.panels(data[1:4],bg=c("green","orange","purple")[data$NSPF],
             pch=21+as.numeric(data$NSPF),main="Fisher data by NSPF",hist.col="red") 
#pairs.panels(data[1:4],bg=c("red","yellow","blue")[data$Tendency],
#  pch=21+as.numeric(data$Tendency),main="Fisher data by Tendency",hist.col="red") 
#to show changing the diagonal

#to show 'significance'
pairs.panels(data[1:4],bg=c("pink","red","blue")[data$NSPF],
             pch=21+as.numeric(data$NSPF),main="Fisher data by NSPF",hist.col="yellow",stars=TRUE) 

#pairs.panels(data[1:4],bg=c("red","yellow","blue")[data$Tendency],
#            pch=21+as.numeric(data$Tendency),main="Fisher data by Tendency",hist.col="red",stars=TRUE) 
#bo6x plot
data %>% ggplot(aes(x=NSPF, y=LB, fill= NSPF)) +
  geom_boxplot() +
  ggtitle("BoxPlot")

#bo6x plot
data %>% ggplot(aes(x=NSPF, y=ASTV, fill= NSPF)) +
  geom_boxplot() +
  ggtitle("BoxPlot")
# Frequency histogram | Display the distribution of the the "NSPF" by "LB de vaisseaux principaux"
data %>% ggplot(aes(x=LB, fill=NSPF, color=NSPF)) +
  geom_histogram(binwidth = 1) +
  labs(title="Nombre de vaisseaux principaux Distribution by NSPF")
# Frequency histogram | Display the distribution of the "NSPF" by "FM"
data %>% ggplot(aes(ASTV, colour = NSPF)) +
  geom_freqpoly(binwidth = 1) +
  labs(title = "ASTV distribution by NSPF")

# Frequency  histogram | Display the distribution of the "NSPF" by "LB"
data %>% ggplot(aes(LB, colour = NSPF)) +
  geom_freqpoly(binwidth = 1) +
  labs(title = "LB distribution by NSPF")
#density plot
data %>%
  ggplot(aes(x=MSTV,fill=NSPF))+
  geom_density(alpha=0.8,color='black')+
  ggtitle("density Plot")
#density plot
data %>%
  ggplot(aes(x=Max,fill=NSPF))+
  geom_density(alpha=0.8,color='black')+
  ggtitle("density Plot")
#density plot
data %>%
  ggplot(aes(x=Mode,fill=NSPF))+
  geom_density(alpha=1,color='black')+
  ggtitle("density Plot")
#Data Partition
set.seed(1234)
pd <-sample(2 ,nrow(data), replace= T, prob=c(0.7,0.3))
train <-data[pd==1,]
test <- data[pd==2,]
library(party)
#create an object pd for decision tree 
pd <-ctree(NSPF~ ., data=train)
#To see plot
plot(pd)
attributes(pd)
pd$confusion
#we can prune the tree by controlling some parameters to make it smaller
# and less complicated
#Predication applying pd model in test data
library(caret)
trainp <- predict(pd,train)
#To check first 6 values of P
head(trainp)
#To compare it with NSP in train dataset
head(train$NSP)
confusionMatrix(trainp, train$NSP)
#here accuracy is 100% b'coz model already saw train data
#Prediction & confusion matrix
testp <- predict(pd,test)
head(testp)
head(test$NSP)
confusionMatrix(testp, test$NSP)
#Accurancy 96%
plot(pd)
tab <- table(testp, test$NSPF)
print(tab)
pd1 <- rpart(NSPF~ ., train)
library(rpart.plot)
rpart.plot(pd1)
#if u want extra stuff in plot
rpart.plot(pd1,extra=1)
#output 1=normal, 2=suspect, 3=pathologic (cardiac issue)
#rpart.plot(dt1,extra=4)
# will show o/p in prob
###Prediction
testp1<-predict(pd1,test)
#we will use the tree model to calculate the misclassification in data
#Misclassification error for train data
#Misclassification error for train data is
1-sum(diag(tab)/sum(tab))
#10%
#Misclassification error with test data
testpred <- predict(pd,newdata=test)
tab <- table(testpred, test$NSPF)
print(tab)
#Misclassification error for train data is
1-sum(diag(tab)/sum(tab))
#10
predicted = predict(pd1, test, type = "class")
testp <- predict(pd, test)
confusionMatrix(testp, test$NSP)
table_mat <- table(test$NSPF, predicted)
table_mat
accuracy_Test <- sum(diag(table_mat)/sum(table_mat))
print(paste('Accuracy for test', accuracy_Test))
#Accuracy=91%
##############################################################
#RandomForest model
#Data Partition
#random seed so we can make the analysis repeatable
set.seed(123)
#Use ind object to divide sample into 70:30 ratio
ind <- sample(2, nrow(data), replace= TRUE, prob=c(0.7,0.3))
train <- data[ind==1,]
test <- data[ind==2,]

#Random Forest package
library(randomForest)
#To make it repeatable use random seed
set.seed(1234)
randomf <- randomForest(NSPF~., data=train)
print(randomf)

# To check attributes of Random Forest model
attributes(randomf)
#To select confusion Matrix
randomf$confusion

#Like this we can use any attributes of random Forest with $ sign
#Prediction & Confusion Matrix
#CFM in caret package
library(caret)
p1 <- predict(randomf, train)
#To check first 6 values of P
head(p1)
#To compare it with NSP in train dataset
head(train$NSP)
confusionMatrix(p1, train$NSP)
# here accuracy is 100% b'coz model already saw train data
#Prediction & confusion matrix
#on Test Data
pd2<- predict(randomf, test)
confusionMatrix(pd2, test
                $NSP)
#Accuracy 93%- because this test data is not seen by RFM

#Error rate of Random Forest
plot(randomf)
#As the number of tree increases #
#This Out of bag errors initially drops down then getting into constant
# So after 300 trees we are not able to improve error
# Tune Random Forest Model
#Tune mtry
t <- tuneRF(train[,-12], train[,12],
            stepFactor = 0.5,
            plot=TRUE,
            ntreeTry=300,
            trace=TRUE,
            improve=0.05)
#You can see OOB error rate reduced at mtry 8
#So this gives an idea to which mtry value to choose for improvement
# try mtry & trees by changing default
randomf <- randomForest(NSPF~., data=train,
                        ntree=300,
                        mtry=8,
                        importance=TRUE,
                        proximity=TRUE)
print(randomf)
#OOB reduced from 6.61 to 6.55 
#No of nodes for the trees
hist(treesize(randomf),
     main=" No of Nodes for the Trees",
     col="blue")
# No of trees in 300 trees we have
#To see which variable place importance in model
varImpPlot(randomf)




