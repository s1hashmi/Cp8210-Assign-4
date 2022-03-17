# Natural Language Processing - Random Forrest
# X===========================================X


# Importing the dataset
dataset_original = read.delim('musical.tsv', quote = '', stringsAsFactors = FALSE)
dataset_original <- musical2
# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')

library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower)) # treat capital-&-lower case as lowercase
corpus = tm_map(corpus, removeNumbers) # remove that is a standard number
corpus = tm_map(corpus, removePunctuation) # remove everything that isn't a standard number or letter.
corpus = tm_map(corpus, removeWords, stopwords()) # Stopwords are unhelpful words like 'i', 'is', 'at', 'me', 'our'.
corpus = tm_map(corpus, stemDocument) # reduce the number of inflectional forms of words appearing in the text. For example, words such as "argue", "argued", "arguing", "argues" are reduced to their common stem "argu".
corpus = tm_map(corpus, stripWhitespace)

# Cloud also remove the word related to what you are predicting e.g. The word 
#   'cloth' is removed because this dataset is on clothing review, so this word will
#   not add any predictive power to the model.

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.99)
dataset = as.data.frame(as.matrix(dtm))
dataset$Score = dataset_original$Score


# Encoding the target feature as factor
dataset$Score = factor(dataset$Score, levels = c(0, 1))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Score, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-720],
                          y = training_set$Score,
                          ntree = 10)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-720])


# Making the Confusion Matrix
cm = table(test_set[, 720], y_pred)

classifier
y_pred
cm

Accuracy = (76 + 70) / (76 + 70 + 31 + 23)
Precision = 76 / (76 + 31)
Recall = 76 / (76 + 23)
F1_Score <- 2 * Precision * Recall / (Precision + Recall)

Accuracy
Precision
Recall
F1_Score

