import numpy as np

class BatchTraining:
    """
    Batch training allows user to save memory by recycling the same variables.
    Testing NB is memory intensive due ot the sparse matrix
    """

    def __init__(self, X_test: np.array, y_test: np.array, vectorizer, batch_size=1000):
        """
        :param X_test: dataset to test
        :param y_test: true labels
        :param vectorizer: vectorizer (SKL TFIDF or CounvtVect)
        :param batch_size: size of training every time
        """
        self.X = X_test
        self.y = y_test
        self.batch_size = batch_size
        self.vectorizer = vectorizer

    def __len__(self):
        """
        :return: number of batches to train
        """
        return (np.ceil(len(self.X) / float(self.batch_size))).astype(int)

    def __iter__(self):
        """
        Example:
        >>> bt = BatchTraining(X,y,vec)
        ... for X, y in bt:
        :return: (X_test_batch, y_test_batch)
        """
        for idx in range(len(self)):
            batch_x = self.vectorizer.transform(
                self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
            ).toarray()
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
            yield batch_x, batch_y