library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(inspectdf)
library(h2o)
library(stopwords)

# 1---
data <- fread("nlpdata.csv")
data %>%  dim()
data %>% colnames()
data %>% glimpse()
data %>% inspect_na()

data$V1 <-  data$V1 %>% as.character()

# Spliting data---
set.seed(123)
split <- data$Liked %>% sample.split(SplitRatio = 0.85)
train <- data %>% subset(split == T)
test <- data %>% subset(split == F)


# Tokenizing-----
it_train <- train$Review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$V1,
         progressbar = F) 


# 1 Modeling ------
vocab <- it_train %>% create_vocabulary()
vocab %>% arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10)

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train),train$V1)

glmnet_classifier <- dtm_train %>% cv.glmnet(y=train[['Liked']],
                                             family='binomial',
                                             type.measure = 'auc',
                                             nfolds = 10,
                                             thresh=0.001,
                                             maxit=1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("=Max AUC")                                             

it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% itoken(ids=test$V1,
progressbar= F)

dtm_test <- it_test %>% create_dtm(vectorizer)
  
pred <- predict(glmnet_classifier,dtm_test,type='response')[,1]  
glmnet:::auc(test$Liked,pred) %>% round(2)


# 2 Modeling----
stop_words <- c(stopwords())

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

prune_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 5,
                   doc_proportion_max =0.5,
                   doc_proportion_min =0.001)

prune_vocab %>% arrange(desc(term_count)) %>% tail(10)

vectorizer <- prune_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y=train[['Liked']],
            family='binomial',
            type.measure = 'auc',
            nfolds = 4,
            thresh=0.001,
            maxit=1000)

glmnet_classifier$cvm %>% max() %>% round(2) %>% paste("=Max AUC")


dtm_test <- it_test %>% create_dtm(vectorizer)
pred <- predict(glmnet_classifier,dtm_test,type='response')[,1]
glmnet:::auc(test$Liked,pred) %>% round(2)


# 3 Modeling------
vocab <- it_train %>% create_vocabulary(ngram = c(1L,2L))

vocab <- vocab %>% 
  prune_vocabulary(term_count_min =3,
                   doc_proportion_max = 0.5)

bigram_vectorizer <- vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(bigram_vectorizer)

dtm_train %>% dim()                                     

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y=train[['Liked']],
            family='binomial',
            type.measure = 'auc',
            nfolds =4,
            thresh=0.001,
            maxit=1000)

glmnet_classifier$cvm %>% max() %>% round(2) %>% paste("= Max AUC")

dtm_test <- it_test %>%create_dtm(bigram_vectorizer) 
pred <- predict(glmnet_classifier,dtm_test,type='response')[,1]
glmnet:::auc(test$Liked,pred) %>% round(2)

