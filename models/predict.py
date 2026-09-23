"""Inference utilities for the Drug Discovery Virtual Screening models.

Generated automatically by `notebook/02_machine_learning_models.ipynb`.
The preprocessing functions below are the EXACT source used during training, so training and
inference cannot drift apart.

Usage
-----
    from predict import ScreeningModel

    model = ScreeningModel()                       # loads everything from this folder
    result = model.predict(raw_dataframe)          # raw columns in, predictions out

Command line
------------
    python predict.py path/to/compounds.csv predictions.csv
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import LabelEncoder

ARTEFACT_DIR = Path(__file__).resolve().parent

PHYSICAL_BOUNDS = {
    'molecular_weight': (1.0, None),
    'polar_surface_area': (0.0, None),
    'h_bond_donors': (0.0, None),
    'h_bond_acceptors': (0.0, None),
    'rotatable_bonds': (0.0, None),
    'protein_length': (1.0, None),
    'protein_pi': (0.0, 14.0),
    'hydrophobicity': (0.0, 1.0),
    'binding_site_size': (0.0, None),
}


def clip_to_physical_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    """Clip descriptors to the range that is chemically possible. Missing values stay missing."""
    out = frame.copy()
    for column, (low, high) in PHYSICAL_BOUNDS.items():
        if column in out.columns:
            out[column] = out[column].clip(lower=low, upper=high)
    return out


def exceeds(series: pd.Series, limit: float) -> pd.Series:
    """1.0 when the descriptor breaks the limit, 0.0 when it does not, NaN when it is unknown.

    A plain `series > limit` silently returns False for a missing value, which would quietly record
    "rule passed" for a compound whose descriptor was never measured. Propagating the NaN instead
    lets the pipeline's imputer handle it like every other gap.
    """
    return (series > limit).astype(float).where(series.notna())


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features to a raw drug-screening table.

    All eleven ligand and protein descriptors must be present; the function works on a single
    compound as well as on the full training table, and never touches either target column.
    Missing values are propagated rather than silently treated as zero.
    """
    out = frame.copy()
    eps = 1e-6

    # ligand-side chemistry
    out["hbond_total"] = out["h_bond_donors"] + out["h_bond_acceptors"]
    out["hba_to_hbd_ratio"] = out["h_bond_acceptors"] / (out["h_bond_donors"] + 1)
    out["tpsa_over_mw"] = out["polar_surface_area"] / (out["molecular_weight"] + eps)
    out["flexibility_index"] = out["rotatable_bonds"] / (out["molecular_weight"] / 100 + eps)
    out["clogp_minus_logp"] = out["compound_clogp"] - out["logp"]
    out["logp_squared"] = out["logp"] ** 2

    # compound / protein interaction terms (the dominant signal in the EDA)
    out["logp_x_pi_exact"] = out["logp"] * out["protein_pi"]
    out["logp_x_hydrophobicity"] = out["logp"] * out["hydrophobicity"]
    out["lipophilic_site_fit"] = out["logp"] * out["binding_site_size"]
    out["mw_per_residue_exact"] = out["molecular_weight"] / (out["protein_length"] + eps)
    out["site_per_residue"] = out["binding_site_size"] / (out["protein_length"] + eps)

    # medicinal-chemistry rules (NaN-aware, see `exceeds`)
    lipinski_violations = (
        exceeds(out["molecular_weight"], 500)
        + exceeds(out["logp"], 5)
        + exceeds(out["h_bond_donors"], 5)
        + exceeds(out["h_bond_acceptors"], 10)
    )
    veber_violations = (
        exceeds(out["rotatable_bonds"], 10)
        + exceeds(out["polar_surface_area"], 140)
    )
    out["lipinski_violations"] = lipinski_violations
    # Multiplying the two 0/1 indicators is a NaN-propagating logical AND.
    out["drug_like"] = ((lipinski_violations <= 1).astype(float).where(lipinski_violations.notna())
                        * (veber_violations == 0).astype(float).where(veber_violations.notna()))

    return out


def safe_label_transform(encoder: LabelEncoder, values: pd.Series, unknown: int = -1) -> np.ndarray:
    """Encode `values`, mapping any category unseen during training to `unknown` instead of raising."""
    lookup = {category: index for index, category in enumerate(encoder.classes_)}
    return values.astype(str).map(lookup).fillna(unknown).astype(int).to_numpy()


class ScreeningModel:
    """Loads the saved artefacts and turns raw compound/protein rows into predictions."""

    OUTPUT_COLUMNS = ["activity_probability", "predicted_active", "predicted_binding_affinity"]

    def __init__(self, artefact_dir: Path | str = ARTEFACT_DIR, check_versions: bool = True):
        self.dir = Path(artefact_dir)
        with open(self.dir / "model_metadata.json", encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        self.classifier = joblib.load(self.dir / "best_classifier.joblib")
        self.regressor = joblib.load(self.dir / "best_regressor.joblib")
        self.label_encoders = joblib.load(self.dir / "label_encoders.joblib")
        self.feature_order = self.metadata["contract"]["model_feature_order"]
        self.threshold = self.metadata["classification"]["decision_threshold"]
        if check_versions:
            self._check_sklearn_version()

    def _check_sklearn_version(self) -> None:
        """Warn loudly when the runtime differs from the version the pickles were built with."""
        built_with = self.metadata.get("library_versions", {}).get("scikit_learn")
        if built_with and built_with != sklearn.__version__:
            warnings.warn(
                f"These artefacts were built with scikit-learn {built_with} but scikit-learn "
                f"{sklearn.__version__} is installed. Unpickling across versions can change "
                f"behaviour silently. See requirements.txt in this folder.",
                RuntimeWarning, stacklevel=3)

    def build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply exactly the training-time preprocessing to a raw table."""
        missing = [c for c in self.metadata["contract"]["required_raw_columns"]
                   if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required raw columns: {missing}")

        out = clip_to_physical_bounds(frame)
        out = engineer_features(out)
        for column, encoder in self.label_encoders.items():
            out[f"{column}_encoded"] = safe_label_transform(encoder, out[column])
        return out[self.feature_order].astype(float)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return activity probability, activity call and predicted binding affinity."""
        if len(frame) == 0:                       # an empty batch is a normal pipeline input
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS, index=frame.index)

        features = self.build_features(frame)
        probability = self.classifier.predict_proba(features)[:, 1]
        affinity = self.regressor.predict(features)
        result = pd.DataFrame({
            "activity_probability": probability,
            "predicted_active": (probability >= self.threshold).astype(int),
            "predicted_binding_affinity": affinity,
        }, index=frame.index)
        for column in ("compound_id", "protein_id"):
            if column in frame.columns:
                result.insert(0, column, frame[column].values)
        return result


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    source = Path(argv[1])
    destination = Path(argv[2]) if len(argv) > 2 else source.with_name(source.stem + "_scored.csv")
    model = ScreeningModel()
    predictions = model.predict(pd.read_csv(source))
    predictions.to_csv(destination, index=False)
    print(f"Wrote {len(predictions):,} predictions to {destination}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
