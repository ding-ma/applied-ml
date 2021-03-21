from abc import ABC, abstractmethod
from utils.activation import ActivationFunction, RUN_DATE
import joblib
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix


class AbstractMLP(ABC):
    def __init__(
        self,
        input_dim,
        output_dim,
        output_fnc,
        learn_rate_init=0.01,
        batch_size=4,
        reg_lambda=1e-2,
        anneal=True,
        num_epochs=50,
        L2=False,
    ):
        self.learn_rate_init = learn_rate_init
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.anneal = anneal
        self.num_epochs = num_epochs
        self.L2 = L2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_fnc: ActivationFunction = output_fnc

        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.file_name = ""
        self.delta = None

    def fit(self, train_array, train_labels_array, x_test=None, y_test=None):
        self.num_iterations = int(train_array.shape[0] / self.batch_size)
        indices = np.arange(train_array.shape[0])

        for epoch in range(self.num_epochs):
            current_index = 0
            running_loss = []

            # shuffle labels between epochs
            np.random.shuffle(indices)
            train_array = train_array[indices]
            train_labels_array = train_labels_array[indices]

            # TODO: shuffle data
            for i in range(1, self.num_iterations + 1):
                X = train_array[current_index : current_index + self.batch_size]
                y = train_labels_array[current_index : current_index + self.batch_size]

                if current_index + 2 * self.batch_size <= train_array.shape[0]:
                    current_index += self.batch_size
                else:  # return to beginning
                    current_index = 0

                num_data = X.shape[0]

                # Annealing schedule
                if epoch < self.num_epochs / 5:
                    learn_rate = self.learn_rate_init
                elif epoch < self.num_epochs / 2:
                    learn_rate = self.learn_rate_init / 10
                elif epoch < 4 * self.num_epochs / 5:
                    learn_rate = self.learn_rate_init / 100
                else:
                    learn_rate = self.learn_rate_init / 1000

                self.forward_prop(X)
                self.backward_pop(num_data, X, y, learn_rate)

                if i % 500 == 0:
                    loss = self.compute_loss(X, y)
                    running_loss.append(loss)
                    logging.info(f"Loss {loss} at epoch {epoch} iteration {i}")

            loss = sum(running_loss) / len(running_loss)
            self.loss_history.append(loss)

            train_acc = self.compute_acc(self.predict(train_array), train_labels_array)
            self.train_acc_history.append(train_acc)

            test_acc = self.compute_acc(self.predict(x_test), y_test)
            self.test_acc_history.append(test_acc)

            epoch_stats = f"""
            Epoch {epoch}
            Avg Loss: {loss}
            Train acc: {train_acc}
            Test acc: {test_acc}
            """
            logging.info(epoch_stats)

    @abstractmethod
    def compute_loss(self, X, y):
        pass

    @abstractmethod
    def forward_prop(self, X):
        pass

    @abstractmethod
    def backward_pop(self, num_data, X, y, learn_rate):
        pass

    def compute_acc(self, y_true, y_pred):
        assert len(y_true) == len(y_pred)

        return 100 * np.sum(y_pred == y_true) / y_pred.shape[0]

    def predict(self, x):
        predictions = []
        for i in x:
            self.forward_prop(i)
            predictions.append(np.argmax(self.delta, axis=1))
        return np.array(predictions).flatten()

    def save(self):
        save_path = Path().cwd().joinpath("pickles").joinpath(self.file_name + ".pkl")
        joblib.dump(self, save_path, compress=1)

    def __confusion_matrix(self, y_true, y_pred, normalize=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Source: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
        """
        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        cm = confusion_matrix(y_true, y_pred)

        f = plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        cm = Path().cwd().joinpath("plots").joinpath(self.file_name + "_cm.png")
        plt.tight_layout()
        plt.savefig(cm)
        f.clear()
        plt.close(f)

    def plot(self, y_true, y_pred):
        logging.info("Making and Saving plots")

        df = pd.DataFrame({"loss": self.loss_history})

        f = plt.figure()
        sns.lineplot(data=df)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss over Epochs")
        plt.tight_layout()
        loss_plot = Path().cwd().joinpath("plots").joinpath(self.file_name + "_loss.png")
        plt.savefig(loss_plot)
        f.clear()
        plt.close(f)

        df = pd.DataFrame({"train": self.train_acc_history, "test": self.test_acc_history})

        f = plt.figure()
        sns.lineplot(data=df)
        plt.xlabel("epochs")
        plt.ylabel("Acc")
        plt.title("Train and Test Accuracy over Epochs")
        plt.tight_layout()
        acc = Path().cwd().joinpath("plots").joinpath(self.file_name + "_acc.png")
        plt.savefig(acc)
        f.clear()
        plt.close(f)

        self.__confusion_matrix(y_true, y_pred)
