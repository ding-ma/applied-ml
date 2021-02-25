# Tracker

## 5-CrossValidation
* Done without Lammetization 


|Model                            |Accuracy|Error  |Params                                                    |
|---------------------------------|--------|-------|----------------------------------------------------------|
|logistic regression              |0.90436 |0.09564|max_itr=12000, solver=sag, vect=CountVectorizer, tol=0.001|
|Multinomial NB (self implemented)|0.868   |0.132  |vect=TfidfVectorizer                                      |
|Bernouilli NB (self implemented) |0.87882 |0.12118|vect=TfidfVectorizer                                      |
|Multinomial NB (sklearn)         |0.87816 |0.12184|vect=TfidfVectorizer                                      |
|Bernouilli NB (sklearn)          |0.86296 |0.13704|vect=CountVectorizer                                      |



### Twenty News

|Model                            |Accuracy|Error  |Params                                                    |
|---------------------------------|--------|-------|----------------------------------------------------------|
|logistic regression              |In progress|In progress|In progress                                               |
|Multinomial NB (self implemented)|0.670808223|17.04542762|vect=TfidfVectorizer                                      |
|Bernouilli NB (self implemented) |0.118125571|48.01189101|vect=TfidfVectorizer                                      |
|Multinomial NB (sklearn)         |0.680995984|16.60193371|vect=TfidfVectorizer                                      |
|Bernouilli NB (sklearn)          |0.441739225|23.82703331|Both                                                      |




## Custom Train size CV
We randomly sampled 5 times in the dataset for each training iteration. Between each interations, some datapoints might be used more than once for taining/testing.
But overall it should provide a good picture of the accuracy at each training size. 

### IMDB
|Train Size|logistic regression (accuracy) |logistic regression (error) |Bernouilli NB (accuracy)|Bernouilli NB (error)|
|----------|-------------------|-------------------|-------------|-------------|
|0.2       |0.88104            |0.11896            |0.831985     |0.168015     |
|0.4       |0.89252            |0.10748            |0.868713333  |0.131286667  |
|0.6       |0.89976            |0.10024            |0.87731      |0.12269      |
|0.8       |0.90634            |0.09366            |0.87948      |0.12052      |

### Twenty News
|Train Size|logistic regression (accuracy)|logistic regression (error)|Multinomial NB (accuracy)|Multinomial NB (error)|
|----------|------------------------------|---------------------------|-------------------------|----------------------|
|0.2       |                              |                           |0.576454202              |19.51556676           |
|0.4       |                              |                           |0.637778564              |18.5126636            |
|0.6       |                              |                           |0.663810029              |17.11430088           |
|0.8       |                              |                           |0.678906872              |16.45709737           |


## Future Ideas if time permits
* Try with lammetization
Just thinking to run with the best results shown above as we are restricted in time.