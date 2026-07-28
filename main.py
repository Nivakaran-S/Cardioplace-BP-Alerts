import sys

from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == '__main__':
    try:
        logging.info("Starting the Cardioplace BP Alerts training pipeline")
        artifact = TrainingPipeline().run_pipeline()

        print("\n=== training pipeline complete ===")
        print(f"bundle          : {artifact.bundle_file_path}")
        print(f"model           : {artifact.trained_model_file_path}")
        print(f"reports         : {artifact.report_dir}")
        print(f"selected family : {artifact.selected_family}")
        if artifact.rule_engine_artifact:
            r = artifact.rule_engine_artifact
            print(f"rule engine     : {r.alerts_fired:,} alerts over {r.rows_evaluated:,} "
                  f"readings | {r.evaluable_rules} evaluable / {r.blocked_rules} blocked rules")
        if artifact.safety_gate_artifact:
            g = artifact.safety_gate_artifact
            print(f"safety gates    : {g.gates_passed}/{g.gates_total} pass | "
                  f"{g.critical_failures} critical failures | promotable={g.promotable}")
    except Exception as e:
        raise CustomException(e, sys)
