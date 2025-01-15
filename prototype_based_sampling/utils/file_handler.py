import os
import pickle


def load_results_from_pkl(path):
    with open(f"{path}/results.pkl", "rb") as file:
        overall_results = pickle.load(file)
    return overall_results


def save_results_to_pkl(overall_results, path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/results.pkl", "wb") as file:
        pickle.dump(overall_results, file)
