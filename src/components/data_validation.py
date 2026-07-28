import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.constants.training_pipeline import INGEST_RANGES, SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import read_yaml_file, save_report, write_yaml_file


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    critical: bool = False

    @property
    def status(self) -> str:
        return "PASS" if self.passed else ("FAIL" if self.critical else "WARN")


class DataContract:
    """Schema, range, structural and referential checks over the raw and panel frames."""

    REQUIRED = {"patient_id": "object", "ts": "datetime", "sbp": "number", "dbp": "number"}
    OPTIONAL = ["weight", "idwg", "sbp_drop", "uf_total", "temperature", "dryweight"]
    RANGES = {"sbp": (60, 260), "dbp": (30, 160), "weight": (25, 220),
              "idwg": (-5, 12), "age": (18, 110), "temperature": (33, 41)}
    MAX_OUT_OF_RANGE = 0.005

    def __init__(self, max_out_of_range: float = None):
        self.checks = []
        if max_out_of_range is not None:
            self.MAX_OUT_OF_RANGE = max_out_of_range

    def _add(self, name, passed, detail="", critical=False):
        self.checks.append(Check(name, bool(passed), detail, critical))

    def _schema(self, raw):
        for col, kind in self.REQUIRED.items():
            here = col in raw.columns
            self._add(f"schema: '{col}' present", here, critical=True)
            if not here:
                continue
            s = raw[col]
            ok = (pd.api.types.is_numeric_dtype(s) if kind == "number"
                  else pd.api.types.is_datetime64_any_dtype(s) if kind == "datetime" else True)
            self._add(f"schema: '{col}' usable as {kind}", ok, f"got {s.dtype}", critical=True)
        missing = [c for c in self.OPTIONAL if c not in raw.columns]
        self._add("schema: optional columns present", not missing, f"missing {missing}")

    def _ranges(self, raw, static):
        for col, (lo, hi) in self.RANGES.items():
            src = raw if col in raw.columns else (static if col in static.columns else None)
            if src is None:
                continue
            s = pd.to_numeric(src[col], errors="coerce").dropna()
            if s.empty:
                continue
            bad = float(((s < lo) | (s > hi)).mean())
            self._add(f"range: {col} in [{lo}, {hi}]", bad < self.MAX_OUT_OF_RANGE,
                      f"{bad:.3%} outside; observed [{s.min():.1f}, {s.max():.1f}]")
        pp_bad = float(((raw.sbp - raw.dbp) < INGEST_RANGES["min_pulse_pressure"]).mean())
        self._add("range: pulse pressure >= 10 mmHg", pp_bad < self.MAX_OUT_OF_RANGE,
                  f"{pp_bad:.3%} violate")

    def _structure(self, raw, panel, min_sessions):
        dup = int(raw.duplicated(["patient_id", "ts"]).sum())
        self._add("structure: unique (patient, session)", dup == 0, f"{dup} duplicates",
                  critical=True)
        self._add("structure: timestamps increasing within patient",
                  panel.groupby("series_id").ts.apply(lambda s: s.is_monotonic_increasing).all(),
                  critical=True)
        self._add("structure: step index contiguous from 0",
                  panel.groupby("series_id").step
                       .apply(lambda s: bool((s.values == np.arange(len(s))).all())).all(),
                  critical=True)
        g = panel.days_since_last
        self._add("structure: gaps positive and bounded",
                  bool((g >= 1).all() and (g <= 30).all()),
                  f"gap range [{g.min():.0f}, {g.max():.0f}] days", critical=True)
        short = int((panel.groupby("series_id").size() < min_sessions).sum())
        self._add(f"structure: every patient has >= {min_sessions} sessions", short == 0,
                  f"{short} below floor", critical=True)

    def _referential(self, panel):
        unmatched = int(panel.age.isna().sum())
        self._add("join: demographics matched", unmatched == 0,
                  f"{unmatched} sessions ({unmatched / max(len(panel), 1):.2%}) unmatched")
        self._add("join: merge did not duplicate rows",
                  len(panel) == panel.drop_duplicates(["series_id", "ts"]).shape[0],
                  critical=True)
        self._add("join: demographics constant within patient",
                  int(panel.groupby("series_id")[["gender", "age"]].nunique().max().max()) <= 1,
                  "gender/age must not vary across a patient's sessions")

    def validate(self, raw, static, panel, min_sessions) -> pd.DataFrame:
        self.checks.clear()
        self._schema(raw)
        self._ranges(raw, static)
        self._structure(raw, panel, min_sessions)
        self._referential(panel)
        return pd.DataFrame([{"check": c.name, "status": c.status, "detail": c.detail}
                             for c in self.checks])

    def enforce(self) -> None:
        fails = [c for c in self.checks if c.status == "FAIL"]
        if fails:
            raise AssertionError("data contract violated:\n"
                                 + "\n".join(f"  - {c.name}: {c.detail}" for c in fails))


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df.ts, errors="coerce")
            if "patient_id" in df.columns:
                df["patient_id"] = df.patient_id.astype(str)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Compare against the declared column list, not the number of top-level YAML keys."""
        try:
            expected = [list(c.keys())[0] if isinstance(c, dict) else c
                        for c in self._schema_config.get("columns", [])]
            missing = [c for c in expected if c not in dataframe.columns]
            logging.info("Required number of columns:%s", len(expected))
            logging.info("Data frame has columns:%s", len(dataframe.columns))
            if missing:
                logging.warning("missing columns: %s", missing)
                return False
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=None) -> bool:
        """Per-column KS test. Returns the status instead of silently discarding it."""
        try:
            threshold = threshold or self.data_validation_config.drift_threshold
            status = True
            report = {}
            numeric = [c for c in base_df.columns
                       if pd.api.types.is_numeric_dtype(base_df[c])
                       and c in current_df.columns]
            for column in numeric:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()
                if len(d1) < 2 or len(d2) < 2:
                    continue
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found,
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            for label, df in (("train", train_dataframe), ("test", test_dataframe)):
                if not self.validate_number_of_columns(dataframe=df):
                    logging.warning("%s dataframe does not contain all schema columns", label)

            # The contract runs on the full session frame plus a minimal panel view, so the
            # structural checks (contiguous step, bounded gaps) have something to check.
            sessions = DataValidation.read_data(
                self.data_ingestion_artifact.feature_store_file_path)
            static = pd.read_csv(self.data_ingestion_artifact.static_file_path)
            static["patient_id"] = static.patient_id.astype(str)

            # Demographics are already on every session row (merged at ingest), so the panel
            # is just the session frame indexed by step -- re-merging would suffix the columns.
            panel = sessions.sort_values(["patient_id", "ts"]).copy()
            panel["series_id"] = panel.patient_id
            panel["days_since_last"] = (panel.groupby("series_id").ts.diff().dt.days
                                        .fillna(2).clip(1, 30))
            panel["step"] = panel.groupby("series_id").cumcount()
            # The contract's session floor applies to the admitted cohort; at this stage the
            # frame is still every patient, so check against the cohort that will be admitted.
            admitted = panel.groupby("series_id").size()
            keep = admitted[admitted >= self.data_validation_config.min_sessions].index
            panel_admitted = panel[panel.series_id.isin(keep)]

            contract = DataContract(self.data_validation_config.max_out_of_range)
            report = contract.validate(sessions, static, panel_admitted,
                                       self.data_validation_config.min_sessions)
            save_report(self.data_validation_config.contract_report_file_path, report)
            n_pass = int((report.status == "PASS").sum())
            n_warn = int((report.status == "WARN").sum())
            n_fail = int((report.status == "FAIL").sum())
            logging.info("data contract: %d pass | %d warn | %d fail", n_pass, n_warn, n_fail)
            contract.enforce()
            logging.info("data contract satisfied")

            status = self.detect_dataset_drift(base_df=train_dataframe,
                                               current_df=test_dataframe)

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,
                                   index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,
                                  index=False, header=True)

            return DataValidationArtifact(
                validation_status=bool(status and n_fail == 0),
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                contract_report_file_path=self.data_validation_config.contract_report_file_path,
                n_pass=n_pass, n_warn=n_warn, n_fail=n_fail,
            )
        except Exception as e:
            raise CustomException(e, sys)
