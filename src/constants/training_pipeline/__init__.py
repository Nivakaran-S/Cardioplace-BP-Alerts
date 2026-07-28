import os

"""
Defining common constant variable for training pipeline.

Values mirror the reference notebook (notebooks/bp_poc_reference_pipeline_v9.ipynb).
Anything under GOVERNANCE is a clinician / ops input: it is never searched, never tuned,
and never derived from the data.
"""

PIPELINE_NAME: str = "CardioplaceBPAlerts"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "sessions.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

SEED: int = 42

"""
Raw corpus (HEMOBP). The three CSVs ship with the repository; vip.csv is tracked with Git LFS.
"""

RAW_DATA_DIR: str = "data"
VIP_FILE_NAME: str = "vip.csv"
D1_FILE_NAME: str = "d1.csv"
IDP_FILE_NAME: str = "idp.csv"

# vip.csv is ~250 MB / ~4.4M readings; it is streamed rather than loaded whole.
INGEST_CHUNKSIZE: int = 1_000_000

# Physiological admission ranges, applied at ingest. Documented here so the filter is
# reviewable rather than buried in a boolean expression.
INGEST_RANGES: dict = {"sbp": (60, 260), "dbp": (30, 160), "min_pulse_pressure": 10}

# Published figures from the HEMOBP data descriptor, used for the ingest reconciliation.
PAPER_COUNTS: dict = {"bp_recordings": 4_366_298, "sessions": 165_986, "patients": 1_075}

# Reference year used to turn birth year into age, per the notebook.
AGE_REFERENCE_YEAR: int = 2016

# Header names that would indicate a pulse column; HEMOBP's published schema has none.
PULSE_CANDIDATES: tuple = ("pulse", "hr", "heart_rate", "heartrate", "pulse_rate", "pr", "bpm")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_STATIC_FILE_NAME: str = "static.csv"
DATA_INGESTION_RECON_FILE_NAME: str = "ingest_reconciliation.csv"

# Cohort admission
MIN_SESSIONS: int = 60
MIN_SPAN_DAYS: int = 120
MAX_PATIENTS = 150  # runtime cap; set to None to use every eligible patient

# Splits: temporal tail per patient, plus a hash-based patient-level holdout.
VAL_FRAC: float = 0.20
TEST_FRAC: float = 0.20
HOLDOUT_PATIENT_FRAC: float = 0.30
PATIENT_SPLIT_SALT: str = "patient-split"

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_CONTRACT_REPORT_FILE_NAME: str = "data_contract.csv"
DATA_VALIDATION_DRIFT_THRESHOLD: float = 0.05
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

# Fraction of values allowed outside a documented physiological range before the check fails.
MAX_OUT_OF_RANGE: float = 0.005

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_PANEL_FILE_NAME: str = "panel.parquet"
DATA_TRANSFORMATION_FEATURE_FILE_NAME: str = "features.parquet"
DATA_TRANSFORMATION_FEATURE_LIST_FILE_NAME: str = "feature_names.yaml"
DATA_TRANSFORMATION_FEATURE_DICT_FILE_NAME: str = "feature_dictionary.csv"
DATA_TRANSFORMATION_LEAKAGE_REPORT_FILE_NAME: str = "leakage_audit.csv"

# Causal feature construction
HORIZONS: tuple = (1, 2, 3)  # sessions ahead (~2, 5, 7 days at 3x/week)
LAGS: tuple = (1, 2, 3, 5, 7, 14)
WINDOWS: tuple = (3, 7, 14, 30)
SIGNALS: tuple = ("sbp", "dbp", "idwg")
EWM_ALPHAS: tuple = (0.1, 0.3)

"""
Model Trainer related content start with MODEL_TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_REPORT_DIR: str = "reports"
MODEL_TRAINER_BUNDLE_FILE_NAME: str = "predictor.joblib"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05

# Row caps so a full run stays tractable on CPU.
MAX_TRAIN_ROWS: int = 20_000
MAX_EVAL_ROWS: int = 5_000

# Random search over the model registry
TUNE: bool = True
TUNE_DRAWS: int = 8
TUNE_FOLDS: int = 3

# Serving
LATENCY_BUDGET_MS: float = 200.0
COLD_START_MIN_READINGS: int = 7
STEADY_STATE_READINGS: int = 48

"""
GOVERNANCE: clinician / ops inputs. Never searched, never tuned, asserted throughout.
"""

POPULATION_THRESHOLD_MMHG: float = 140.0
EMERGENCY_FLOOR_MMHG: float = 180.0  # never personalised
OFFSET_CAP_LOOSEN: float = 15.0      # asymmetric on purpose: loosening is the hazardous way
OFFSET_CAP_TIGHTEN: float = 25.0
ALERT_BUDGET_PCT: float = 5.0        # staffing capacity, not a statistic
WARN_WINDOW: int = 3                 # sessions of lead time the detector must deliver
EVENT_QUANTILE: float = 0.95         # personal quantile defining a high-BP event

GOVERNANCE_KEYS: list = [
    "population_threshold_mmHg",
    "emergency_floor_mmHg",
    "offset_cap_loosen",
    "offset_cap_tighten",
    "alert_budget_pct",
    "warn_window",
    "event_quantile",
]

# Offset blend defaults (searched over warm/k/q; the caps above are deliberately excluded).
OFFSET_WARM: int = 48
OFFSET_K: float = 30.0
OFFSET_Q: float = 0.90
OFFSET_SEARCH_WARM: tuple = (24, 36, 48, 72)
OFFSET_SEARCH_K: tuple = (10, 30, 60)
OFFSET_SEARCH_Q: tuple = (0.85, 0.90, 0.95)

# Advisory arbitration
CONFIDENCE_TAU: float = 0.55

# Fairness gate: a subgroup may not be worse than overall by more than this margin.
FAIR_MARGIN_MMHG: float = 1.5
