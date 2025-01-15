"""
A simple file that regenerates the confusion matrix based on the results reported to tensorboard.
"""

import math
import os

import numpy as np
from tbparse import SummaryReader

#################
# parse data
#################

base_dir = "../tb_data_final_results/"

all_bwts = {}

# we take the average over the last 5 test accuracies
# that is the mean over the values at idx [95, 100], [195, 200], [295, 300]
enable_avg_over_last_5 = False

# loop over all folders in base dir with name pattern date__*
for folder in os.listdir(base_dir):

    if (
        "results" in folder
        or "logs" in folder
        or "crash" in folder
        or "first_repetition" in folder
    ):
        continue

    try:
        method_name = folder.split("__")[-1]
        print(f"\nProcess: Method: {method_name}")

        tb_run = os.path.join(base_dir, folder)
        reader = SummaryReader(tb_run)
        df = reader.scalars

        # get stream with name "Top1_Acc_Exp/eval_phase/test_stream/Task000"
        bwt_exp = df.loc[
            df["tag"].str.startswith("StreamBWT/eval_phase/test_stream"),
            "value",
        ].values

        print(f"Calc BWT from stream of Task000")

        mean_btws = []
        for task in range(2, 3):

            index_range = task * 100 + 95 if enable_avg_over_last_5 else task * 100 + 99
            raw_bwts = (
                bwt_exp[index_range : index_range + 5].mean().mean()
                if enable_avg_over_last_5
                else bwt_exp[index_range]
            )

            # [index_range : index_range + 5].mean()
            print(f" - Mean BWT for Task{task}: {raw_bwts:.3f}")
            mean_btws.append(raw_bwts)

        mean_bwt = np.array(mean_btws).mean()
        print(f"» Mean BWT: {mean_bwt:.3f}")

        if method_name not in all_bwts:
            all_bwts[method_name] = []

        if math.isnan(mean_bwt):
            continue

        all_bwts[method_name].append(mean_bwt)

    except Exception as e:
        print(f"Error: {e}")


print("\n\n++++++++++++++++++++++\nAll BWTs:")

for method_name, accs in all_bwts.items():

    # Assuming accs is your array
    accs = np.array(accs)

    # Drop NaN values
    accs = accs[~np.isnan(accs)]

    # Calculate statistics
    mean = accs.mean()
    std = accs.std()
    length = len(accs)

    print(
        f"Method: {method_name}, Mean BWT: {mean:.3f} +/- {np.array(accs).std():.3f} with rep={length}"
    )
