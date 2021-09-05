import math
import os

import dataframe_image as dfi
import pandas as pd

INPUT_DIR = "/Users/damian.baciur/polibuda/magis/runner/outputs/{}/summary"
MAIN_INPUT = "/Users/damian.baciur/polibuda/magis/runner/outputs/{}/summary/summary.csv"
# OUTPUT = "/Users/damian.baciur/polibuda/magis/runner/outputs/{}/summary/summary.png"

OUTPUT_DIR = "/Users/damian.baciur/polibuda/magis/runner/outputs/{}/summary"


def convert(run_name):
    input_dir = INPUT_DIR.format(run_name)

    for file in os.listdir(input_dir):

        if ".csv" not in file:
            continue
        if file == "summary.csv":
            continue

        base_name = file[:-4]
        input_path = f"{input_dir}/{base_name}.csv"
        output_path = f"{input_dir}/{base_name}.png"

        df = pd.read_csv(input_path, converters={
            "Acc": str,
            "Prec 0": str,
            "Prec 1": str,
            "F1 0": str,
            "F1 1": str
        })

        # trunc = lambda x: math.trunc(1000 * x) / 1000
        # for col in ["It.", "Acc", "Prec 0", "Prec 1", "Prec 0", "F1 0", "F1 1"]:
        #     df[col] = df[col].apply(trunc)

        df_styled = df.style.background_gradient()
        dfi.export(df_styled, output_path, max_rows=-1)


def divide(run_name):
    groups = ["alex", "alex_deep", "resnet", "vgg", "vgg_deep"]
    rename_dict = {
        "alex": "alex_m2",
        "alex_deep": "alex",
        "resnet": "resnet",
        "vgg": "vgg_m2",
        "vgg_deep": "vgg"
    }
    input_file = MAIN_INPUT.format(run_name)
    output_dir = OUTPUT_DIR.format(run_name)

    df = pd.read_csv(input_file)

    for group in groups:
        group_df = df[df["Model"] == group]
        del group_df["Model"]

        group_df.reset_index(drop=True, inplace=True)
        group_df = group_df.rename(columns={"Iteration": "It."})

        trunc = lambda x: math.trunc(1000 * x) / 1000
        for col in ["Acc", "Prec 0", "Prec 1", "Prec 0", "F1 0", "F1 1"]:
            group_df[col] = group_df[col].apply(trunc)

        renamed = rename_dict[group]
        output_path = f"{output_dir}/{renamed}.csv"
        group_df.to_csv(output_path, index=False)


def to_latex(run_name):
    groups = ["alex", "alex_deep", "resnet", "vgg", "vgg_deep"]
    rename_dict = {
        "alex": "AlexNetM2",
        "alex_deep": "AlexNet",
        "resnet": "ResNet",
        "vgg": "VggM2",
        "vgg_deep": "Vgg"
    }
    input_file = MAIN_INPUT.format(run_name)
    df = pd.read_csv(input_file)

    trunc = lambda x: math.trunc(1_000_00 * x) / 1_000_00
    # trunc = lambda x: x

    print()

    prev_it = -1
    for _, row in df.iterrows():
        name = rename_dict[row["Model"]]
        it = row["Iteration"]
        set = row["Set"]
        acc = trunc(row["Acc"])
        prec0 = trunc(row["Prec 0"])
        prec1 = trunc(row["Prec 1"])
        f1_0 = trunc(row["F1 0"])
        f1_1 = trunc(row["F1 1"])

        if it != prev_it:
            print("\\hline")
            prev_it = it
        latex = f"{name} & {it} &  {set} & {acc} & {prec0} & {prec1} & {f1_0} & {f1_1} \\\\"
        print(latex)


if __name__ == "__main__":
    result_dirs = [
        "samples_websites_10s_pure_weight_none",
        "samples_websites_10s_pure_weights_1_1",
        "samples_websites_10s_pure_weights_1_50",
        "samples_websites_10s_weight_none",
        "samples_websites_10s_weights_1_1",
        "samples_websites_10s_weights_1_50"
    ]
    for result_dir in result_dirs:
        # divide(f"websites/{result_dir}")
        # convert(f"websites/{result_dir}")
        pass
    to_latex(f"websites/samples_websites_10s_pure_weight_none")
