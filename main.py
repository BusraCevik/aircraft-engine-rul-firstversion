import os

from src.data_loader import run_data_loader
from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_pipeline
from src.visualization import run_visualization_pipeline


# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_PATH = os.path.join(DATA_DIR, "raw", "train_FD001.txt")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURED_DIR = os.path.join(DATA_DIR, "featured")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")


# -------------------------------
# MAIN
# -------------------------------
def main():

    # data loader
    run_data_loader(
        raw_path=RAW_PATH,
        processed_dir=PROCESSED_DIR
    )

    # preprocessing
    run_preprocessing_pipeline(
        processed_dir=PROCESSED_DIR
    )

    # feature engineering
    run_feature_pipeline(
        processed_dir=PROCESSED_DIR,
        featured_dir=FEATURED_DIR
    )

    # visualization (processed kullanıyoruz)
    run_visualization_pipeline(
        processed_dir=PROCESSED_DIR,
        output_dir=FIGURES_DIR
    )


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()