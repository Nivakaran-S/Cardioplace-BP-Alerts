import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from src.exception.custom_exception import CustomException
from src.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_dataframe(file_path: str, df: pd.DataFrame) -> str:
    """Persist a frame as Parquet, falling back to CSV when no parquet engine is installed.

    Column names matter downstream -- the rule engine and the serving path both index the
    feature frame by name -- so this never degrades to a bare numpy array.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            df.to_parquet(file_path, index=False)
            return file_path
        except Exception as exc:
            fallback = os.path.splitext(file_path)[0] + ".csv"
            logging.warning("parquet write failed (%s); wrote %s instead", exc,
                            os.path.basename(fallback))
            df.to_csv(fallback, index=False)
            return fallback
    except Exception as e:
        raise CustomException(e, sys) from e


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Read a frame written by `save_dataframe`, honouring the CSV fallback."""
    try:
        if file_path.endswith(".parquet") and os.path.exists(file_path):
            return pd.read_parquet(file_path)
        csv_path = os.path.splitext(file_path)[0] + ".csv"
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        raise FileNotFoundError(f"neither {file_path} nor {csv_path} exists")
    except Exception as e:
        raise CustomException(e, sys) from e


def save_report(file_path: str, df: pd.DataFrame) -> str:
    """Write a scorecard to CSV. Reports replace the notebook's `display()` calls."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        raise CustomException(e, sys) from e


def stable_bucket(key: str, salt: str = "", buckets: int = 100) -> int:
    """Deterministic hash bucket -- stable across runs, machines and row order."""
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) % buckets


def deterministic_patient_split(ids, holdout_frac: float,
                                salt: str = "patient-split") -> dict:
    """Assign each patient to fit/holdout by hash, not by shuffle order."""
    cut = int(round(holdout_frac * 100))
    return {i: ("holdout" if stable_bucket(str(i), salt) < cut else "fit") for i in ids}


def frame_digest(df: pd.DataFrame) -> str:
    """Content hash of a frame -- order-insensitive on columns, sensitive on values."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df.reindex(sorted(df.columns), axis=1),
                                   index=True).values.tobytes()).hexdigest()[:12]


def summarise(df: pd.DataFrame, name: str) -> None:
    logging.info("%s: %s rows x %s cols | %.1f MB | digest %s",
                 name, f"{len(df):,}", df.shape[1],
                 df.memory_usage(deep=True).sum() / 1e6, frame_digest(df))
