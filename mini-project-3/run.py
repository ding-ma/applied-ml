from model.MLP import MLP, Layer, Softmax, ReLU
from utils.utils import evaluate_acc, RUN_DATE
import logging
import sys
from datetime import datetime
import inspect
from preprocess.get_data import aquire_data
from itertools import islice
import numpy as np
# make sure this dictionary has the same variable names as the constructor of MLP


data_preprocess_params = {
    "threshold": True,
    "normalize": True,
    "augment_data": {"rotate": True, "shift": True, "zoom": True, "shear": True, "all": True},
}

if __name__ == "__main__":
    
    # turn lambda into string
    # params_to_log = MLP_params
    # params_to_log["activation_fnc"] = ":".join(
    #     inspect.getsourcelines(MLP_params["activation_fnc"])[0][0].strip().split(":")[1:]
    # )

    experiment_description = f"""
    some description about your experiment

    Experiment Parameters
    

    Preprocess Parameters
    {data_preprocess_params}
    """

    file_name = ""  # optional


    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # TODO: uncooment when ready to run real tests
            # logging.FileHandler(filename=f"logs/{RUN_DATE}-{file_name}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


    logging.info(experiment_description)

    # TODO: add kFold CV
    train, test = aquire_data(**data_preprocess_params)



    train_slice = list(islice(train, 1000))
    x_train = np.array([i[0] for i in train_slice])
    y_train = np.array([i[1] for i in train_slice]).reshape(-1, 1)


    test_slice = list(islice(test, 100))
    x_test = np.array([i[0] for i in test_slice])
    y_test = np.array([i[1] for i in test_slice]).reshape(-1, 1)

    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)



    mlp = MLP()

    input_layer = Layer(28*28, 256, ReLU())
    hidden_layer = Layer(28*28, 10, Softmax())

    mlp.add_layer(input_layer)
    mlp.add_layer(hidden_layer)


    mlp.fit(x_train, y_train, 1000)

    y_pred = mlp.predict(x_test)
    logging.info(f"accuracy {evaluate_acc(y_pred, y_test)}")
