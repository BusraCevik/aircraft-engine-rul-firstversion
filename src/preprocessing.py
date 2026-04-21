import pandas as pd
import os


# -------------------------------
# ADD RUL remaining useful life
# -------------------------------
def add_rul(df):
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]

    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop("max_cycle", axis=1, inplace=True)

    return df


# def add_rul(df):
#
#     # sort (safe)
#     df = df.sort_values(["unit_id", "cycle"])
#
#     # max cycle
#     max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
#     max_cycles.columns = ["unit_id", "max_cycle"]
#
#     # merge
#     df = df.merge(max_cycles, on="unit_id")
#
#     # RUL
#     df["RUL"] = df["max_cycle"] - df["cycle"]
#
#     # optional: clip
#     df["RUL"] = df["RUL"].clip(upper=130)
#
#     # drop temp
#     df.drop("max_cycle", axis=1, inplace=True)
#
#     return df

# -------------------------------
# RUN PREPROCESSING PIPELINE
# -------------------------------
def run_preprocessing_pipeline(processed_dir):
    input_path = os.path.join(processed_dir, "train_processed.csv")

    df = pd.read_csv(input_path)
    df = add_rul(df)

    df.to_csv(input_path, index=False)