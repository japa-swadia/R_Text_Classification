################################--SER 599 Thesis--###############################
#-------------------------------------------------------------------------------#
#--Version 1.0 : Japa Swadia--#
#--Classification of requirements statements using k nearest neighbors classification--#

#Load data set for training
#Read csv file containing Requirement statement and Requirement Category columns
load_train <- read.csv("C:/Users/japas_000/Documents/Thesis/model/train_data_small.csv", header = TRUE, nrows = 672)

#Install and load tm package for text mining
library(tm)
#Load required packages
library(class) # KNN model
library(SnowballC) # Stemming words
library(ggplot2) # Plotting data
library(kknn) #Weighted knn

#Prepare data frame for training
train_df <- as.data.frame(load_train[ , -1])

#Create corpus of requirements statements
reqs <- Corpus(VectorSource(train_df$Requirement.Statement))

# Data Visualization for entire data set
tdf <- data.frame(table(train_df$Requirement.Type))
ggplot(tdf, aes(x = tdf$Var1, y = tdf$Freq)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.1f%%", tdf$Freq/672 * 100)), 
            vjust = -.5)


#Clean the corpus
reqs <- tm_map(reqs, stripWhitespace) #remove white spaces
reqs <- tm_map(reqs, content_transformer(tolower)) #transform to lower case
reqs <- tm_map(reqs, removeWords, stopwords("english")) #remove stopwords
reqs <- tm_map(reqs, removeNumbers) #remove numbers
reqs <- tm_map(reqs, removePunctuation) #remove punctuations
reqs <- tm_map(reqs, stemDocument, language = "english") #stem words

#Create Document Term Matrix 
req_dtm <- DocumentTermMatrix(reqs)
sparse_req_dtm <- removeSparseTerms(req_dtm, sparse= 0.97) #remove sparse terms

#Transform the dtm into a data frame
req_df <- as.data.frame(data.matrix(sparse_req_dtm), stringsAsFactors = FALSE)

#Bind Requirements Category column to the data frame (known classification)
req_df <- cbind(req_df, train_df$Requirement.Type)

#Name this column Category
colnames(req_df)[ncol(req_df)] <- "Category"

#Divide data set into equal row samples, each corresponding to training data and test data
train_set <- sample(1:nrow(req_df), 332)
test_set <- (1:nrow(req_df))[- train_set]

# Isolate classifier
classifier <- req_df[, "Category"]

# Create model data and remove "category"
model_data <- req_df[,!colnames(req_df) %in% "Category"]

# Create model: training set, test set, training set classifier
#k=5
knn.pred <- knn(model_data[train_set, ], model_data[test_set, ], classifier[train_set], k=5)
#k=1
knn1.pred <- knn1(model_data[train_set, ], model_data[test_set, ], classifier[train_set])
#kknn.pred <- kknn(classifier[train_set]~., model_data[train_set, ], model_data[test_set, ], distance = 2, 
#                  kernel = "rank" )
#summary(kknn.pred)
#fit <- fitted(kknn.pred)
#kknn.tab <- table( fit)

# Confusion matrix
conf.mat <- table("Predictions" = knn.pred, Actual = classifier[test_set])
conf.mat1 <- table("Predictions" = knn1.pred, Actual = classifier[test_set])

# Find error rate and parameters
error.rate <- (sum(conf.mat) - diag(conf.mat))/sum(conf.mat)
tp <- conf.mat[1,1] #true positives
fp <- conf.mat[2,1] #false positives
tn <- conf.mat[2,2] #true negatives
fn <- conf.mat[1,2] #false negatives

tp1 <- conf.mat1[1,1] #true positives
fp1 <- conf.mat1[2,1] #false positives
tn1 <- conf.mat1[2,2] #true negatives
fn1 <- conf.mat1[1,2] #false negatives

# Calculate Recall
recall <- tp/(tp+fn) * 100
recall1 <- tp1/(tp1+fn1)*100

#Calculate Precision
precision <- tp/(tp+fp) * 100
precision1 <- tp1/(tp1+fp1) * 100

# Calculate Accuracy of model
accuracy <- sum(diag(conf.mat))/length(test_set) * 100
accuracy1 <- sum(diag(conf.mat1))/length(test_set) * 100

# Create data frame with test data and predicted category
output.pred <- cbind(knn.pred, model_data[test_set, ])
output2.pred <- cbind(knn1.pred, model_data[test_set, ])

# Produce a csv file of predicted output
write.csv(output.pred, file="C:/Users/japas_000/Documents/Thesis/model/knn_output.csv")
write.csv(output2.pred, file="C:/Users/japas_000/Documents/Thesis/model/knn_output2.csv")

#End of model