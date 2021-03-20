from model.MLP import MLP, Layer, Softmax, Relu
from utils.utils import evaluate_acc, RUN_DATE
import logging
import sys
from datetime import datetime
import inspect
from preprocess.get_data import aquire_data


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
    {params_to_log}

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

    input_layer = Layer(28*28, 256, Relu())
    hidden_layer = Layer(256, 10, Softmax())