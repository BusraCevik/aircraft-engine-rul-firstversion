import pandas as pd
import os


# -------------------------------
# LOAD DATA
# -------------------------------
def load_nasa_data(file_path):
    columns = ["unit_id", "cycle"] + \
              [f"setting_{i}" for i in range(1, 4)] + \
              [f"sensor_{i}" for i in range(1, 22)]

    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.dropna(axis=1)
    df.columns = columns

    return df


# -------------------------------
# RUN DATA LOADER
# -------------------------------
def run_data_loader(raw_path, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

    df = load_nasa_data(raw_path)

    output_path = os.path.join(processed_dir, "train_processed.csv")
    df.to_csv(output_path, index=False)