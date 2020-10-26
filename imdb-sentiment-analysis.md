IMDB Sentiment Analysis
================

# Sentiment Analysis for movie reviews

## Part 1: Loading and wrangling

1)  Loading the data

<!-- end list -->

``` r
set.seed(12345)
imdb <- read.csv("Data/IMDB.csv")
colnames(imdb)[1] <- 'review_text' 
imdb$review_text <- as.character(imdb$review_text)
```

2)  Training the
classifier

<!-- end list -->

``` r
#Removing stop words, whitespaces, numbers & words occuring in less than 5% of the documents
#stem the terms
#lowercase
# 1/0 for occurence

matrix <- create_matrix(imdb$review_text, language="english", removeSparseTerms = 0.95,
                        removeStopwords=TRUE, removeNumbers=TRUE, stemWords=TRUE, 
                        stripWhitespace=TRUE, toLower=TRUE)
mat <- as.matrix(matrix)

test_instn = sample(nrow(imdb),0.3*nrow(imdb))
test_x = mat[test_instn,]
train_x = mat[-test_instn,]
test_y <- imdb[test_instn, 2]
train_y <- imdb[-test_instn, 2]

#Modelling and predictions
model <- maxent(train_x, train_y)
preds <- predict(model, test_x, type = 'response')
class_performance(table(test_y, as.factor(preds[,1]),dnn=c("Actual", "Predicted")))[1] * 100
```

    ## [1] 81.33333

Training the maxent classifier gives an accuracy of 81.12%

3)  Let’s look at some of the wrongly classified

<!-- end list -->

1.  Instance 209, contains ‘This movie succeeds at being one of the most
    unique movies youve seen. However this comes from the fact that you
    cant make heads or tails of this mess. It almost seems as a series
    of challenges set up to determine whether or not you are willing to
    walk out of the movie and give up the money you just paid. If you
    don’t want to feel slight..’ is classified as a P instead of a N
    that could be because of the first highlighted line.

2.  Instance 241, contains ‘This film is basically two hours of Dafoes
    character drinking himself - nearly literally - to death. The only
    surprise in this film is that you didnt have enough clues or
    character knowledge to be surprised. It was just a grim sad waste of
    time.’ Is classified as a P instead of a N because the author goes
    on to describe how great the actors are in a lot of lines.

3.  Instance 562 contains, ‘It doesnt really sound like a logical mix of
    story lines and incoherent but both plot lines blend in perfectly’
    which could through off the classifier.

## Part 2: Improving the performance

1)  Keeping the stopwords in
there

<!-- end list -->

``` r
matrix <- create_matrix(imdb$review_text, language="english", removeSparseTerms = 0.95,
                        removeStopwords=FALSE, removeNumbers=TRUE, stemWords=TRUE, 
                        stripWhitespace=TRUE, toLower=TRUE)

test_instn = sample(nrow(imdb),0.3*nrow(imdb))
test_x = mat[test_instn,]
train_x = mat[-test_instn,]
test_y <- imdb[test_instn, 2]
train_y <- imdb[-test_instn, 2]

model <- maxent(train_x, train_y)
preds <- predict(model, test_x, type = 'response')

class_performance(table(test_y, as.factor(preds[,1]),dnn=c("Actual", "Predicted")))[1] * 100
```

    ## [1] 81.08

We get an accuracy of 81.32% which is slightly better than the previous
one.

2)  Increasing the sparsity to 10% of the
document.

<!-- end list -->

``` r
matrix <- create_matrix(imdb$review_text, language="english", removeSparseTerms = 0.90,
                        removeStopwords=TRUE, removeNumbers=TRUE, stemWords=TRUE, 
                        stripWhitespace=TRUE, toLower=TRUE)

test_instn = sample(nrow(imdb),0.3*nrow(imdb))
test_x = mat[test_instn,]
train_x = mat[-test_instn,]
test_y <- imdb[test_instn, 2]
train_y <- imdb[-test_instn, 2]

#Modelling and Predictions
model <- maxent(train_x, train_y)
preds <- predict(model, test_x, type = 'response')

#Checking the performance with the fucntion
class_performance(table(test_y, as.factor(preds[,1]),dnn=c("Actual", "Predicted")))[1] * 100
```

    ## [1] 82.10667

We get an accuracy of 81.65% Which is a slight improvement
