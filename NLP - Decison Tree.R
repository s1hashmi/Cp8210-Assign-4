# Natural Language Processing - Decison Tree
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
# Fitting Decision Tree Classification to the Training set
# install.packages("parsnip")
library(parsnip)
dt = decision_tree(
  mode = "classification",
  engine = "rpart",
  cost_complexity = NULL,
  tree_depth = NULL,
  min_n = NULL
)

classifier_dt = fit(dt, Score ~ ., data = training_set)

library(caret)
# trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

classifier_dt <- train(x = training_set[-720],
                       y = training_set$Score, method = "rpart")

## OR ##
# classifier_dt <- train(Score ~ training_set[-720], data = training_set, method = "rpart",
#                   parms = list(split = "information"),
#                   trControl=trctrl,
#                   tuneLength = 10)


training_set$Score
str(training_set)

# Predicting the Test set results
y_pred_dt = predict(classifier_dt, newdata = test_set[-720])


# Making the Confusion Matrix
cm_dt = confusionMatrix(y_pred_dt, test_set$Score )  #check accuracy

classifier_dt
y_pred_dt
cm_dt

Accuracy_dt = (96 + 21) / (96 + 21 + 72 + 11)
Precision_dt = 96 / (96 + 72)
Recall_dt = 96 / (96 + 11)
F1_Score_dt = 2 * Precision * Recall / (Precision + Recall)

Accuracy_dt
Precision_dt
Recall_dt
F1_Score_dt


