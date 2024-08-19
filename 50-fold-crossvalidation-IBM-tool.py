import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from models.custom_model import CustomModel
from utils.utils import get_model_path, pretty_names
import utils.load
from tensorflow.keras import backend as K
from utils.vars import CLASSIFICATION, REGRESSION
from utils.save import save_fig
import seaborn as sns
from sklearn.utils import resample
import keras.backend as K

import matplotlib.pyplot as plt
import time


import logging

omicLogger = logging.getLogger("OmicLogger")


def boxplot_scorer_cv(
    experiment_folder,
    model_list: list[str],
    problem_type: str,
    seed_num: int,
    fit_scorer: str,
    scorer_dict,
    data,
    true_labels,
    nsplits=50,  # Change to no groups you need
    group_size=28,  # Number of samples in each group (can be changed to 20% of samples in data)
    save=True,
    holdout=False,
):
    """
    Create a graph of boxplots for all models in the folder, using the specified fit_scorer from the config.

    By default this uses a Monte Carlo Cross-Validation with 50 random groups, each containing 28 samples (20% of samples).
    """
    omicLogger.debug("Creating boxplot_scorer_cv...")
    # Create the plot objects
    fig, ax = plt.subplots()
    # Container for the scores
    all_scores = []
    print(f"Size of data for boxplot: {data.shape}")

    # Seed the random number generator for reproducibility
    np.random.seed(seed_num)

    # Loop over the models
    for model_name in model_list:
        # Load the model if trained
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting boxplot for {model_name} using {fit_scorer}")
        # Select the scorer
        scorer_func = scorer_dict[fit_scorer]
        # Container for scores for this cross-val for this model
        scores = []
        num_testsamples_list = []

        for fold in range(nsplits):
            omicLogger.debug(f"{model_name}, fold {fold}")
            print(f"{model_name}, fold {fold}")

            # Randomly select group_size samples for the current fold
            test_idx = np.random.choice(data.shape[0], group_size, replace=False)
            train_idx = np.setdiff1d(np.arange(data.shape[0]), test_idx)

            # Load the model
            model = utils.load.load_model(model_name, model_path)
            # Handle the custom model
            if isinstance(model, tuple(CustomModel.__subclasses__())):
                # Remove the test data to avoid any saving
                if model.data_test is not None:
                    model.data_test = None
                if model.labels_test is not None:
                    model.labels_test = None

            model.fit(data[train_idx], true_labels[train_idx])
            # Calculate the score
            # Need to take the absolute value because of the make_scorer sklearn convention
            score = np.abs(scorer_func(model, data[test_idx], true_labels[test_idx]))
            num_testsamples = len(true_labels[test_idx])
            # Add the scores
            scores.append(score)
            num_testsamples_list.append(num_testsamples)

        # Maintain the total list
        all_scores.append(scores)
        # Save CV results
        d = {"Scores CV": scores, "Dim test": num_testsamples_list}
        fname = f"{experiment_folder / 'results' / 'scores_CV'}_{model_name}_{fold}"
        fname += "_holdout" if holdout else ""
        df = pd.DataFrame(d)
        df.to_csv(fname + ".csv")

    pretty_model_names = [pretty_names(name, "model") for name in model_list]

    # Make the boxplot
    sns.boxplot(data=all_scores, ax=ax, width=0.4)
    ax.set_xticklabels(pretty_model_names, rotation=90)
    # Format the graph
    ax.set_xlabel("ML Methods")

    fig = plt.gcf()
    # Save the graph
    if save:
        fname = f"{experiment_folder / 'graphs' / 'boxplot'}_{fit_scorer}"
        fname += "_holdout" if holdout else ""
        save_fig(fig, fname)

    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()
