# Tracker

## Regular kFold
* Done Lammetization 

### IMDB
|Model              |Accuracy|Error|Params                                                    |
|-------------------|--------|-----|----------------------------------------------------------|
|logistic regression|90%     |9.50%|max_itr=12000, solver=sag, vect=CountVectorizer, tol=0.001|
|Multinomial NB     |88%     |12%  |vect=TfidfVectorizer                                      |
|Bernouilli NB      |88%     |12%  |vect=TfidfVectorizer                                      |


### Twenty News

|Model              |Accuracy   |Error      |Params              |
|-------------------|-----------|-----------|--------------------|
|logistic regression|In progress|In progress|In progress         |
|Multinomial NB     |67%        |17%        |vect=TfidfVectorizer|
|Bernouilli NB      |12%        |48%        |vect=TfidfVectorizer|



## Custom Train size CV


## With Lammetization
Just thinking to run with the best results shown above as we are restricted in time.