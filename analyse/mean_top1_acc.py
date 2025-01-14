"""
A simple file that regenerates the confusion matrix based on the results reported to tensorboard.
"""

import os

import numpy as np
from tbparse import SummaryReader

#################
# parse data
#################

base_dir = "../tb_data_final_results/"

all_accuracies = {}

# we take the average over the last 5 test accuracies
# that is the mean over the values at idx [95, 100], [195, 200], [295, 300]
enable_avg_over_last_5 = False

# loop over all folders in base dir with name pattern date__*
for folder in os.listdir(base_dir):

    if "results" in folder or "logs" in folder:
        continue

    try:
        method_name = folder.split("__")[-1]
        print(f"\nProcess: Method: {method_name}")

        tb_run = os.path.join(base_dir, folder)
        reader = SummaryReader(tb_run)
        df = reader.scalars

        # get stream with name "Top1_Acc_Exp/eval_phase/test_stream/Task000"
        top1_acc_exp = df.loc[
            df["tag"].str.startswith("Top1_Acc_Exp/eval_phase/test_stream/Task000"),
            "value",
        ].values

        print(f"Calc Top1 Accuracy from stream of Task000")

        mean_accs = []
        for task in range(3):

            index_range = task * 100 + 95 if enable_avg_over_last_5 else task * 100 + 99
            mean_top1_acc = (
                top1_acc_exp[index_range : index_range + 5].mean().mean()
                if enable_avg_over_last_5
                else top1_acc_exp[index_range]
            )

            # [index_range : index_range + 5].mean()
            print(f" - Mean Top1 Accuracy for Task{task}: {mean_top1_acc:.2f}")
            mean_accs.append(mean_top1_acc)

        mean_accuracy = np.array(mean_accs).mean()
        print(f"» Mean Top1 Accuracy: {mean_accuracy:.2f}")

        if method_name not in all_accuracies:
            all_accuracies[method_name] = []

        all_accuracies[method_name].append(mean_accuracy)

    except Exception as e:
        print(f"Error: {e}")

print("\n\n++++++++++++++++++++++\nAll Accuracies:")
for method_name, accs in all_accuracies.items():
    print(
        f"Method: {method_name}, Mean Acc: {np.array(accs).mean():.2f} +/- {np.array(accs).std():.2f}"
    )
