import pandas as pd
import matplotlib.pyplot as plt
import os


# -------------------------------
# LOAD DATA
# -------------------------------
def load_processed_data(processed_dir):
    path = os.path.join(processed_dir, "train_processed.csv")
    return pd.read_csv(path)


# -------------------------------
# CREATE OUTPUT DIR
# -------------------------------
def create_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# -------------------------------
# PLOT SENSOR
# -------------------------------
def plot_sensor(df, sensor_name, output_dir):
    plt.figure()

    for unit in df["unit_id"].unique()[:5]:
        temp = df[df["unit_id"] == unit]
        plt.plot(temp["cycle"], temp[sensor_name])

    plt.xlabel("Cycle")
    plt.ylabel(sensor_name)

    path = os.path.join(output_dir, f"{sensor_name}.png")
    plt.savefig(path)
    plt.close()


# -------------------------------
# RUN VISUALIZATION PIPELINE
# -------------------------------
def run_visualization_pipeline(processed_dir, output_dir):

    # load data
    df = load_processed_data(processed_dir)

    # create output dir
    create_output_dir(output_dir)

    # plot sensors
    for i in range(1, 4):
        plot_sensor(df, f"sensor_{i}", output_dir)