import os
import sys

import pandas as pd

from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.custom_exception import CustomException
from src.logging.logger import logging, timer
from src.utils.main_utils.utils import (save_dataframe, save_object, save_report,
                                        summarise, write_yaml_file)
from src.utils.ml_utils.feature.causal_features import (CausalFeatureBuilder, add_splits,
                                                        build_panel, feature_dictionary,
                                                        leakage_audit)


class DataTransformation:
    """Panel construction, splits and strictly-causal feature engineering.

    There is no fitted preprocessor here by design: imputation lives inside `make_model`
    (median SimpleImputer for the linear pipelines; HistGradientBoosting handles NaN
    natively), so exactly the same transform runs at training and at serving time.
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
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

    def get_data_transformer_object(self) -> CausalFeatureBuilder:
        """The causal feature builder is the transformer for this pipeline."""
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            return CausalFeatureBuilder(self.data_transformation_config)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            cfg = self.data_transformation_config
            # Train and test were split temporally at ingest; features must be built over the
            # whole timeline per patient, then labelled by split. Building them separately
            # would leave the first rows of the test tail with no history to look back at.
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path)
            sessions = (pd.concat([train_df, test_df], ignore_index=True)
                        .drop(columns=["step"], errors="ignore")
                        .sort_values(["patient_id", "ts"]))

            static_cols = ["patient_id", "gender", "age", "DM"]
            static = (sessions[[c for c in static_cols if c in sessions.columns]]
                      .drop_duplicates("patient_id"))
            sessions = sessions.drop(columns=[c for c in ("gender", "age", "DM")
                                              if c in sessions.columns])

            with timer("panel construction"):
                panel = build_panel(sessions, static, cfg)
                panel = add_splits(panel, cfg)
            summarise(panel, "panel")

            fb = self.get_data_transformer_object()
            with timer("feature construction"):
                F = fb.transform(panel)
            features = fb.feature_names_
            summarise(F, "features")
            logging.info("%d features across %d patients", len(features), F.series_id.nunique())

            # The causal contract is the single most consequential thing in this pipeline,
            # so it is tested rather than assumed.
            audit = leakage_audit(F, features, cfg)
            save_report(cfg.leakage_report_file_path, audit)
            failed = audit[audit.status == "FAIL"]
            leakage_clean = failed.empty
            if not leakage_clean:
                raise AssertionError("leakage audit failed:\n"
                                     + failed.to_string(index=False))
            warned = audit[audit.status == "WARN"]
            if len(warned):
                logging.warning("leakage audit: %d probe(s) warned\n%s",
                                len(warned), warned.to_string(index=False))
            logging.info("leakage audit clean: %d probes passed", int((audit.status == "PASS").sum()))

            save_dataframe(cfg.panel_file_path, panel)
            save_dataframe(cfg.feature_file_path, F)
            save_dataframe(cfg.transformed_train_file_path, F[F.split.isin(["train", "val"])])
            save_dataframe(cfg.transformed_test_file_path, F[F.split == "test"])
            save_report(cfg.feature_dictionary_file_path, feature_dictionary(F, features))
            write_yaml_file(cfg.feature_list_file_path,
                            {"features": list(features),
                             "horizons": list(cfg.horizons),
                             "signals": list(cfg.signals)},
                            replace=True)
            # The feature builder holds no fitted state, but the pipeline contract expects an
            # object at this path and the serving bundle reconstructs one from the config.
            save_object(cfg.transformed_object_file_path, fb)

            return DataTransformationArtifact(
                transformed_object_file_path=cfg.transformed_object_file_path,
                transformed_train_file_path=cfg.transformed_train_file_path,
                transformed_test_file_path=cfg.transformed_test_file_path,
                panel_file_path=cfg.panel_file_path,
                feature_file_path=cfg.feature_file_path,
                feature_list_file_path=cfg.feature_list_file_path,
                leakage_report_file_path=cfg.leakage_report_file_path,
                n_features=len(features),
                n_patients=int(F.series_id.nunique()),
                leakage_clean=bool(leakage_clean),
            )
        except Exception as e:
            raise CustomException(e, sys)
