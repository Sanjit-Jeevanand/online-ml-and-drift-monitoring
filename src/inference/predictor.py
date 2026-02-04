from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import json
import numpy as np
import pandas as pd

from src.models.registry import load_model, load_metadata
from src.features.metadata import load_feature_metadata_file as load_feature_contract


# ============================================================
# Predictor
# ============================================================

class Predictor:

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        artifact_path: Optional[Path] = None,
        feature_contract_path: Path = Path("artifacts/features/feature_metadata.json"),
    ) -> None:
        if artifact_path is not None:
            if model_name is not None or model_version is not None:
                raise ValueError("Provide either artifact_path or model_name and model_version, not both.")
            # Load model and preprocessor from artifact path
            model_file = artifact_path / "model.joblib"
            preprocessor_file = artifact_path / "preprocessor.joblib"
            metadata_file = artifact_path / "metadata.json"

            self.model = joblib.load(model_file)
            self.preprocessor = joblib.load(preprocessor_file)

            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)

            self.model_name = self.metadata.get("model_name")
            self.model_version = self.metadata.get("model_version")

            # Shadow artifacts may not have full registry metadata.
            # Do NOT crash inference if these fields are missing.
            if self.model_name is None:
                self.model_name = "unknown"
            if self.model_version is None:
                self.model_version = "unknown"
        else:
            if model_name is None or model_version is None:
                raise ValueError("Must provide either artifact_path or both model_name and model_version.")
            self.model_name = model_name
            self.model_version = model_version

            # ----------------------------------------------------
            # Load model + preprocessor from registry
            # ----------------------------------------------------

            self.model, self.preprocessor = load_model(
                model_name=model_name,
                version=model_version,
            )

            # ----------------------------------------------------
            # Load metadata (lineage + sanity checks)
            # ----------------------------------------------------

            self.metadata = load_metadata(
                model_name=model_name,
                version=model_version,
            )

        # ----------------------------------------------------
        # Load feature contract
        # ----------------------------------------------------

        self.feature_contract = load_feature_contract(feature_contract_path)

        self.feature_names = self.feature_contract.feature_names
        self.feature_types = self.feature_contract.feature_types

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        self._validate_artifact_compatibility()

    # ========================================================
    # Public API
    # ========================================================

    def predict_proba(self, raw_input: Dict[str, Any]) -> float:

        # ----------------------------------------------------
        # Validate + coerce input
        # ----------------------------------------------------

        X = self._prepare_input(raw_input)

        # ----------------------------------------------------
        # Preprocess + predict
        # ----------------------------------------------------

        X_transformed = self.preprocessor.transform(X)

        proba = self.model.predict_proba(X_transformed)[0, 1]

        return float(proba)

    # ========================================================
    # Internal helpers
    # ========================================================

    def _prepare_input(self, raw_input: Dict[str, Any]) -> pd.DataFrame:

        from src.features.contracts import ALL_FEATURES, FORBIDDEN_COLUMNS

        clean_input = {k: v for k, v in raw_input.items() if k not in FORBIDDEN_COLUMNS}

        missing = [f for f in ALL_FEATURES if f not in clean_input]
        extra = [k for k in clean_input.keys() if k not in ALL_FEATURES]

        if missing:
            raise ValueError(f"Missing required features: {missing}")

        if extra:
            raise ValueError(f"Unexpected extra features: {extra}")

        row = {}
        for name in ALL_FEATURES:
            value = clean_input[name]

            # All raw features are numeric in this dataset
            try:
                value = float(value)
            except Exception:
                raise ValueError(
                    f"Feature '{name}' must be numeric; got {value!r}"
                )

            row[name] = value

        df = pd.DataFrame([row], columns=ALL_FEATURES)

        return df

    def _validate_artifact_compatibility(self) -> None:

        contract_version = self.feature_contract.version
        artifact_contract = (
            self.metadata
            .get("feature_contract", {})
            .get("version")
        )

        if artifact_contract is None:
            raise ValueError(
                "Model metadata missing feature contract version."
            )

        if contract_version != artifact_contract:
            raise ValueError(
                "Feature contract mismatch:\n"
                f"  Registry artifact expects: {artifact_contract}\n"
                f"  Local feature contract is: {contract_version}"
            )