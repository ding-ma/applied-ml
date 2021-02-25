# Training Results

## 5-CrossValidation

### With Bigrams

#### IMDB
|Model                            |Accuracy|Error  |Params                                                    |
|---------------------------------|--------|-------|----------------------------------------------------------|
|logistic regression              |0.90436 |0.09564|max_itr=12000, solver=sag, vect=CountVectorizer, tol=0.001|
|Multinomial NB (self implemented)|0.868   |0.132  |vect=TfidfVectorizer                                      |
|Bernouilli NB (self implemented) |0.87882 |0.12118|vect=TfidfVectorizer                                      |
|Multinomial NB (sklearn)         |0.87816 |0.12184|vect=TfidfVectorizer                                      |
|Bernouilli NB (sklearn)          |0.86296 |0.13704|vect=CountVectorizer                                      |


#### Twenty News

|Model                            |Accuracy|Error  |Params                                                    |
|---------------------------------|--------|-------|----------------------------------------------------------|
|logistic regression              |0.7334738713408114|14.832257994537189|max_itr=9000, solver=saga, vect=TfidfVectorizer, tol=0.01   |
|Multinomial NB (self implemented)|0.670808223|17.04542762|vect=TfidfVectorizer                                      |
|Bernouilli NB (self implemented) |0.118125571|48.01189101|vect=TfidfVectorizer                                      |
|Multinomial NB (sklearn)         |0.680995984|16.60193371|vect=TfidfVectorizer                                      |
|Bernouilli NB (sklearn)          |0.441739225|23.82703331|vect=TfidfVectorizer and vect=CountVectorizer             |

* logistic regression grid search took over 30h+!


### With Stemmed words and bigrams

#### IMDB
|Model                           |Accuracy|Error  |Params                                                    |
|--------------------------------|--------|-------|----------------------------------------------------------|
|logistic regression             |0.89758 |0.10242|max_itr=12000, solver=sag, vect=CountVectorizer, tol=0.001|
|Bernouilli NB (self implemented)|0.8698  |0.1302 |vect=TfidfVectorizer                                      |

#### Twenty News
|Model                            |Accuracy   |Error      |Params              |
|---------------------------------|-----------|-----------|--------------------|
|logistic regression              |    0.89758      | 0.10242           | max_itr=12000, solver=sag, vect=CountVectorizer, tol=0.001                   |
|Multinomial NB (self implemented)|0.670968163|17.25376865|vect=TfidfVectorizer|


### With Stemmed words without bigrams

#### IMDB
|Model                           |Accuracy|Error  |Params                                                    |
|--------------------------------|--------|-------|----------------------------------------------------------|
|logistic regression             |0.88792 |0.11208|max_itr=12000, solver=sag, vect=CountVectorizer, tol=0.001|
|Bernouilli NB (self implemented)|0.86106 |0.13894|vect=TfidfVectorizer                                      |


#### Twenty News
|Model                            |Accuracy   |Error      |Params              |
|---------------------------------|-----------|-----------|--------------------|
|logistic regression              |  0.733102575         |  14.86355119         | max_iter=9000, solver=saga, vect=TfidfVectorizer, tol=0.01                   |
|Multinomial NB (self implemented)|0.707474781|15.70895216|vect=TfidfVectorizer|


## Custom Train Size CV
We randomly sampled 5 times in the dataset for each training iteration. Between each interations, some datapoints might be used more than once for taining/testing.
But overall it should provide a good picture of the accuracy at each training size. 

#### IMDB
* Logistic Regression Params: `max_itr=12000, solver=sag, vect=CountVectorizer, tol=0.001`
* Naive Bayes Params: `vect=TfidfVectorizer`

|Train Size|logistic regression (accuracy) |logistic regression (error) |Bernouilli NB (accuracy)|Bernouilli NB (error)|
|----------|-------------------|-------------------|-------------|-------------|
|0.2       |0.88104            |0.11896            |0.831985     |0.168015     |
|0.4       |0.89252            |0.10748            |0.868713333  |0.131286667  |
|0.6       |0.89976            |0.10024            |0.87731      |0.12269      |
|0.8       |0.90634            |0.09366            |0.87948      |0.12052      |

#### Twenty News

* Logisitc Regression Params: `max_iter=9000, solver=saga, vect=TfidfVectorizer, tol=0.01`
* Naive Bayes Params: `vect=TfidfVectorizer`

|Train Size|logistic regression (accuracy)|logistic regression (error)|Multinomial NB (accuracy)|Multinomial NB (error)|
|----------|------------------------------|---------------------------|-------------------------|----------------------|
|0.2       |0.668063938                   |17.36167673                |0.576454202              |19.51556676           |
|0.4       |0.706667846                   |16.39943403                |0.637778564              |18.5126636            |
|0.6       |0.722499337                   |15.6628018                 |0.663810029              |17.11430088           |
|0.8       |0.737543115                   |14.43268772                |0.678906872              |16.45709737           |
