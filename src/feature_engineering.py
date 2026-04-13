import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# LOAD
def load_processed_data(processed_dir):
    return pd.read_csv(os.path.join(processed_dir, "train_processed.csv"))


# SAVE
def save_featured_data(df, featured_dir):
    os.makedirs(featured_dir, exist_ok=True)
    df.to_csv(os.path.join(featured_dir, "train_featured.csv"), index=False)


# LOW VARIANCE (SAFE)
def remove_low_variance(df, threshold=0.0):
    sensor_cols = [c for c in df.columns if "sensor_" in c]

    variances = df[sensor_cols].var()
    selected = variances[variances > threshold].index.tolist()

    # 🔥 önemli: ID ve target kaybolmasın
    keep_cols = selected + ["unit_id", "cycle", "RUL"]

    return df[keep_cols]


# NORMALIZATION
def normalize(df):
    sensor_cols = [c for c in df.columns if "sensor_" in c]

    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    return df


# SMOOTHING
def add_smoothing(df, alpha=0.3):
    sensor_cols = [c for c in df.columns if "sensor_" in c]

    new_cols = {}
    for col in sensor_cols:
        new_cols[f"{col}_smooth"] = df.groupby("unit_id")[col].transform(
            lambda x: x.ewm(alpha=alpha).mean()
        )

    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


# ROLLING
def add_rolling_features(df, window=15):
    sensor_cols = [c for c in df.columns if "sensor_" in c]

    new_cols = {}

    for col in sensor_cols:
        new_cols[f"{col}_mean"] = df.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        new_cols[f"{col}_std"] = df.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


# DIFF
def add_diff_features(df):
    sensor_cols = [c for c in df.columns if "sensor_" in c]

    new_cols = {}
    for col in sensor_cols:
        new_cols[f"{col}_diff"] = df.groupby("unit_id")[col].diff()

    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


# HEALTH INDEX (SAFE)
def add_health_index(df):
    sensor_cols = [c for c in df.columns if "sensor_" in c]

    temp = df[sensor_cols].fillna(0)

    scaled = StandardScaler().fit_transform(temp)

    hi = PCA(n_components=1).fit_transform(scaled)

    df["health_index"] = hi

    return df


# CORRELATION
def remove_high_correlation(df, threshold=0.95):
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]

    return df.drop(columns=drop_cols)


# RUN
def run_feature_pipeline(processed_dir, featured_dir):

    df = load_processed_data(processed_dir)

    df = remove_low_variance(df)
    df = normalize(df)
    df = add_smoothing(df)
    df = add_rolling_features(df)
    df = add_diff_features(df)

    df = df.fillna(0)

    df = add_health_index(df)
    df = remove_high_correlation(df)

    save_featured_data(df, featured_dir)