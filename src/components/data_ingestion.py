import os
import sys

import numpy as np
import pandas as pd

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception.custom_exception import CustomException
from src.logging.logger import logging, timer
from src.utils.main_utils.utils import summarise


class HemobpLoader:
    """Streams the local HEMOBP CSVs and reduces them to one row per dialysis session.

    vip.csv is ~250 MB / ~4.4M readings, so it is read in chunks and reduced twice: once
    per chunk, then again across chunks, because a session can straddle a chunk boundary.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.meta = {}

    def _resolve(self, path: str, label: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at {path}. The HEMOBP CSVs ship with this repository; "
                f"vip.csv is tracked with Git LFS, so run `git lfs pull` if it is a pointer file."
            )
        return path

    def _reduce_vip(self) -> pd.DataFrame:
        key = ["pid", "date"]
        dtypes = {"pid": "int64", "measuretime": "int16",
                  "sbp": "int16", "dbp": "int16", "uf": "float32"}
        ranges = self.config.ingest_ranges
        parts, n_raw, n_dropped = [], 0, 0

        path = self._resolve(self.config.vip_file_path, "vip.csv")
        for chunk in pd.read_csv(path, chunksize=self.config.chunksize,
                                 usecols=["pid", "datatime", "measuretime", "sbp", "dbp", "uf"],
                                 dtype=dtypes):
            n_raw += len(chunk)
            chunk["date"] = pd.to_datetime(chunk.datatime, errors="coerce").dt.normalize()
            chunk = chunk.dropna(subset=["date"])
            lo_s, hi_s = ranges["sbp"]
            lo_d, hi_d = ranges["dbp"]
            keep = (chunk.sbp.between(lo_s, hi_s) & chunk.dbp.between(lo_d, hi_d)
                    & ((chunk.sbp - chunk.dbp) >= ranges["min_pulse_pressure"]))
            n_dropped += int((~keep).sum())
            chunk = chunk[keep]
            if chunk.empty:
                continue
            chunk = chunk.sort_values(key + ["measuretime"])
            first = (chunk.drop_duplicates(key, keep="first")[key + ["measuretime", "sbp", "dbp"]]
                     .rename(columns={"measuretime": "mt_first"}))
            last = (chunk.drop_duplicates(key, keep="last")[key + ["measuretime", "sbp"]]
                    .rename(columns={"measuretime": "mt_last", "sbp": "sbp_post"}))
            agg = chunk.groupby(key, as_index=False).agg(
                sbp_min=("sbp", "min"), uf_total=("uf", "max"), n_meas=("sbp", "size"))
            parts.append(first.merge(last, on=key).merge(agg, on=key))
            del chunk

        if not parts:
            raise ValueError("no admissible BP readings survived the ingest filters")

        v = pd.concat(parts, ignore_index=True)
        del parts
        # Second reduction across chunk boundaries. Explicit merges, not positional
        # .values assignment across differently sorted frames.
        f = (v.sort_values(key + ["mt_first"]).drop_duplicates(key, keep="first")
             [key + ["sbp", "dbp"]])
        l = (v.sort_values(key + ["mt_last"]).drop_duplicates(key, keep="last")
             [key + ["sbp_post"]])
        a = v.groupby(key, as_index=False).agg(sbp_min=("sbp_min", "min"),
                                               uf_total=("uf_total", "max"),
                                               n_meas=("n_meas", "sum"))
        s = f.merge(l, on=key).merge(a, on=key)
        s["sbp_drop"] = s.sbp - s.sbp_min
        self.meta.update(raw_bp_rows=n_raw, rows_out_of_range=n_dropped,
                         sessions_derived=len(s))
        return s

    def load(self):
        """Return (sessions, static): one row per (patient, session) plus demographics."""
        s = self._reduce_vip()
        s["pid"] = s.pid.astype(str)

        d1 = pd.read_csv(self._resolve(self.config.d1_file_path, "d1.csv"))
        d1["date"] = pd.to_datetime(d1.keyindate, errors="coerce").dt.normalize()
        d1 = d1.dropna(subset=["date"])
        d1["pid"] = d1.pid.astype(str)
        s = s.merge(d1[["pid", "date", "weightstart", "weightend", "dryweight", "temperature"]]
                    .drop_duplicates(["pid", "date"]), on=["pid", "date"], how="left")
        s["weight"] = s.weightstart
        s["idwg"] = s.weightstart - s.dryweight

        idp = pd.read_csv(self._resolve(self.config.idp_file_path, "idp.csv"))
        idp["pid"] = idp.pid.astype(str)
        idp["age"] = self.config.age_reference_year - idp.birthday
        static = idp[["pid", "gender", "age", "DM"]].rename(columns={"pid": "patient_id"})

        self.meta.update(patients_in_vip=int(s.pid.nunique()),
                         patients_in_idp=int(idp.pid.nunique()))
        return s.rename(columns={"pid": "patient_id", "date": "ts"}), static

    def reconciliation(self, sessions: pd.DataFrame = None) -> pd.DataFrame:
        """Reconcile against the published data descriptor rather than restating it.

        A session here is a (patient, calendar-date) pair surviving the ingest ranges, so the
        sessions ratio is expected to differ from the published figure. A large gap on the
        other two rows would mean an incomplete read, not a definitional difference.
        """
        paper = self.config.paper_counts
        recon = pd.DataFrame([
            dict(quantity="BP recordings (raw rows)", published=paper["bp_recordings"],
                 derived=self.meta.get("raw_bp_rows")),
            dict(quantity="Sessions", published=paper["sessions"],
                 derived=self.meta.get("sessions_derived")),
            dict(quantity="Patients", published=paper["patients"],
                 derived=self.meta.get("patients_in_idp")),
            dict(quantity="Rows dropped out of range", published=np.nan,
                 derived=self.meta.get("rows_out_of_range")),
        ])
        recon["ratio"] = (recon.derived / recon.published).round(3)

        if sessions is not None and len(sessions):
            # DGP caveat, made visible rather than asserted: the upper tail is operationally
            # censored. Collection re-measured any SBP > 200 and kept the LATER value, so no
            # extreme-value claim is supportable from this corpus, and none is made.
            censored = float((sessions.sbp >= 200).mean())
            logging.info("pre-dialysis SBP >= 200 mmHg: %.3f%% of sessions -- the upper tail is "
                         "operationally censored (re-measured, later value kept)", 100 * censored)
            recon = pd.concat([recon, pd.DataFrame([dict(
                quantity="Sessions with SBP >= 200 (censored tail)", published=np.nan,
                derived=round(censored, 5), ratio=np.nan)])], ignore_index=True)
        return recon


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def export_sessions_as_dataframe(self):
        """Read the HEMOBP corpus from the local data directory."""
        try:
            loader = HemobpLoader(self.data_ingestion_config)
            with timer("HEMOBP ingest (vip -> sessions, merge d1 + idp)"):
                sessions, static = loader.load()
            logging.info("ingest meta: %s", loader.meta)
            return sessions, static, loader.reconciliation(sessions)
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self, sessions: pd.DataFrame, static: pd.DataFrame,
                                       recon: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            sessions.to_csv(feature_store_file_path, index=False, header=True)
            static.to_csv(self.data_ingestion_config.static_file_path, index=False, header=True)
            recon.to_csv(self.data_ingestion_config.reconciliation_file_path, index=False)
            return sessions
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, sessions: pd.DataFrame):
        """Per-patient temporal tail, not a random row split.

        A random split puts a patient's future readings in train and their past in test.
        Every metric downstream would be measuring a model that has seen the answer.
        """
        try:
            cfg = self.data_ingestion_config
            df = sessions.sort_values(["patient_id", "ts"]).copy()
            step = df.groupby("patient_id").cumcount()
            last = step.groupby(df.patient_id).transform("max")
            is_test = step > (last * (1 - cfg.test_frac)).round()

            # `step` itself is not written: it is a split helper, and leaving it in would
            # dominate the drift report (test steps are higher than train steps by
            # construction). Data transformation recomputes it on the full timeline.
            train_set = df[~is_test]
            test_set = df[is_test]
            logging.info("Performed per-patient temporal train/test split "
                         "(%d train rows, %d test rows)", len(train_set), len(test_set))

            dir_path = os.path.dirname(cfg.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(cfg.training_file_path, index=False, header=True)
            test_set.to_csv(cfg.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file path.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            sessions, static, recon = self.export_sessions_as_dataframe()
            summarise(sessions, "sessions")
            summarise(static, "static")
            # Demographics ride along on every session row so the train/test CSVs are
            # self-describing. `static` is deduplicated first: a repeated patient_id would
            # multiply session rows and trip the contract's join check.
            sessions = sessions.merge(static.drop_duplicates("patient_id"),
                                      on="patient_id", how="left")
            sessions = self.export_data_into_feature_store(sessions, static, recon)
            self.split_data_as_train_test(sessions)
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                static_file_path=self.data_ingestion_config.static_file_path,
                reconciliation_file_path=self.data_ingestion_config.reconciliation_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
