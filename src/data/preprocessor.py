"""
Data Preprocessing Module
==========================

Handles data cleaning, missing value imputation, outlier removal,
and dataset splitting for the chemical property prediction pipeline.

Features:
    - Missing value handling with domain-aware strategies
    - Outlier detection using IQR and Z-score methods
    - Feature scaling (Standard, MinMax, Robust)
    - Train/validation/test splitting
    - Data quality reports
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.config import settings
from src.utils.exceptions import DataCollectionError
from src.utils.logger import get_data_logger

logger = get_data_logger()


class DataPreprocessor:
    """
    Preprocesses chemical compound data for ML training.

    Attributes:
        scaler_type: Type of feature scaling to apply
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        random_state: Random seed for reproducibility
        feature_scaler: Fitted scaler instance
        target_scalers: Dictionary of fitted target scalers
    """

    SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(
        self,
        scaler_type: str = "standard",
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the preprocessor.

        Args:
            scaler_type: Feature scaling method ('standard', 'minmax', 'robust')
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation set
            random_state: Random seed
        """
        if scaler_type not in self.SCALERS:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Use: {list(self.SCALERS.keys())}")

        self.scaler_type = scaler_type
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.feature_scaler: Optional[object] = None
        self.target_scalers: Dict[str, object] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.metadata: Dict = {}

        logger.info(
            f"DataPreprocessor initialized (scaler={scaler_type}, "
            f"test={test_size}, val={val_size})"
        )

    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load dataset from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df

    def clean_data(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        drop_duplicates: bool = True,
        handle_missing: str = "impute",
        outlier_method: Optional[str] = "iqr",
        outlier_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Clean and preprocess the dataset.

        Args:
            df: Input DataFrame
            required_columns: Columns that must be present
            drop_duplicates: Whether to remove duplicate rows
            handle_missing: Strategy for missing values ('impute', 'drop', 'none')
            outlier_method: Outlier detection method ('iqr', 'zscore', None)
            outlier_threshold: Threshold for outlier detection

        Returns:
            Cleaned DataFrame
        """
        initial_shape = df.shape
        logger.info(f"Cleaning data: {initial_shape[0]} rows, {initial_shape[1]} columns")

        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise DataCollectionError(f"Missing required columns: {missing}")

        # Drop duplicates
        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} duplicate rows")

        # Handle missing values
        if handle_missing == "drop":
            before = len(df)
            df = df.dropna()
            logger.info(f"Dropped {before - len(df)} rows with missing values")
        elif handle_missing == "impute":
            df = self._impute_missing(df)

        # Remove outliers
        if outlier_method:
            df = self._remove_outliers(df, method=outlier_method, threshold=outlier_threshold)

        logger.info(f"Cleaned: {initial_shape} -> {df.shape}")
        return df.reset_index(drop=True)

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using domain-aware strategies.

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()

        # Numeric columns: use median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Imputed {col} with median: {median_val}")

        # Categorical columns: use mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    logger.debug(f"Imputed {col} with mode: {mode_val[0]}")

        return df

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Remove outliers from numeric columns.

        Args:
            df: DataFrame
            method: Outlier detection method
            threshold: Detection threshold

        Returns:
            DataFrame with outliers removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["cid", "charge"]]

        if method == "iqr":
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                before = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                removed = before - len(df)

                if removed > 0:
                    logger.debug(f"Removed {removed} outliers from {col}")

        elif method == "zscore":
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores < threshold
                df = df.iloc[mask[mask].index]

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        smiles_column: str = "smiles",
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare features and targets for ML training.

        Args:
            df: Cleaned DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names
            smiles_column: Name of SMILES column (excluded from scaling)

        Returns:
            Dictionary with X, y, and metadata
        """
        self.feature_columns = feature_columns
        self.target_columns = target_columns

        # Validate columns exist
        all_cols = feature_columns + target_columns
        missing = set(all_cols) - set(df.columns)
        if missing:
            available = set(df.columns) - set(all_cols)
            logger.warning(f"Missing columns: {missing}")
            logger.info(f"Available columns: {available}")
            # Use available columns
            feature_columns = [c for c in feature_columns if c in df.columns]
            target_columns = [c for c in target_columns if c in df.columns]

        # Extract features and targets
        X = df[feature_columns].copy()
        y = df[target_columns].copy() if target_columns else None

        # Store metadata
        self.metadata = {
            "n_samples": len(df),
            "n_features": len(feature_columns),
            "n_targets": len(target_columns),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "smiles_column": smiles_column,
        }

        logger.info(
            f"Features prepared: {X.shape}, Targets: {y.shape if y is not None else None}"
        )

        return {"X": X, "y": y, "smiles": df[smiles_column] if smiles_column in df.columns else None}

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        smiles: Optional[pd.Series] = None,
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Split data into train/validation/test sets.

        Args:
            X: Feature matrix
            y: Target matrix
            smiles: Optional SMILES series

        Returns:
            Dictionary with train/val/test splits
        """
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        if smiles is not None:
            smiles_trainval, smiles_test = train_test_split(
                smiles,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        else:
            smiles_trainval = smiles_test = None

        # Second split: train vs val
        val_fraction = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction,
            random_state=self.random_state,
        )

        if smiles_trainval is not None:
            smiles_train, smiles_val = train_test_split(
                smiles_trainval,
                test_size=val_fraction,
                random_state=self.random_state,
            )
        else:
            smiles_train = smiles_val = None

        logger.info(
            f"Data split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        result = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        if smiles_train is not None:
            result.update({
                "smiles_train": smiles_train,
                "smiles_val": smiles_val,
                "smiles_test": smiles_test,
            })

        return result

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        fit: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Scale features using the configured scaler.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            fit: Whether to fit the scaler on training data

        Returns:
            Dictionary with scaled features
        """
        scaler_class = self.SCALERS[self.scaler_type]

        if fit or self.feature_scaler is None:
            self.feature_scaler = scaler_class()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(
                X_train_scaled, columns=X_train.columns, index=X_train.index
            )
            logger.info(f"Fitted {self.scaler_type} scaler on {X_train.shape[0]} samples")
        else:
            X_train_scaled = self.feature_scaler.transform(X_train)
            X_train_scaled = pd.DataFrame(
                X_train_scaled, columns=X_train.columns, index=X_train.index
            )

        result = {"X_train": X_train_scaled}

        if X_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
            result["X_val"] = pd.DataFrame(
                X_val_scaled, columns=X_val.columns, index=X_val.index
            )

        if X_test is not None:
            X_test_scaled = self.feature_scaler.transform(X_test)
            result["X_test"] = pd.DataFrame(
                X_test_scaled, columns=X_test.columns, index=X_test.index
            )

        return result

    def scale_targets(
        self,
        y_train: pd.DataFrame,
        y_val: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
        target_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Scale target variables. Only scales regression targets.

        Args:
            y_train: Training targets
            y_val: Validation targets
            y_test: Test targets
            target_types: Dictionary mapping target name to 'regression' or 'classification'

        Returns:
            Dictionary with scaled targets
        """
        if target_types is None:
            target_types = {col: "regression" for col in y_train.columns}

        result = {}

        for col in y_train.columns:
            target_type = target_types.get(col, "regression")

            if target_type == "classification":
                # Don't scale classification targets
                result[f"y_train_{col}"] = y_train[[col]]
                if y_val is not None:
                    result[f"y_val_{col}"] = y_val[[col]]
                if y_test is not None:
                    result[f"y_test_{col}"] = y_test[[col]]
                continue

            # Scale regression targets
            scaler = StandardScaler()
            y_train_col = y_train[[col]].values

            y_train_scaled = scaler.fit_transform(y_train_col)
            self.target_scalers[col] = scaler

            result[f"y_train_{col}"] = pd.DataFrame(
                y_train_scaled, columns=[col], index=y_train.index
            )

            if y_val is not None:
                y_val_scaled = scaler.transform(y_val[[col]].values)
                result[f"y_val_{col}"] = pd.DataFrame(
                    y_val_scaled, columns=[col], index=y_val.index
                )

            if y_test is not None:
                y_test_scaled = scaler.transform(y_test[[col]].values)
                result[f"y_test_{col}"] = pd.DataFrame(
                    y_test_scaled, columns=[col], index=y_test.index
                )

        return result

    def inverse_transform_target(self, values: np.ndarray, target_name: str) -> np.ndarray:
        """
        Inverse transform scaled target values back to original scale.

        Args:
            values: Scaled values
            target_name: Name of the target variable

        Returns:
            Values in original scale
        """
        if target_name in self.target_scalers:
            return self.target_scalers[target_name].inverse_transform(
                values.reshape(-1, 1)
            ).flatten()
        return values

    def get_feature_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature statistics.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with feature statistics
        """
        stats = X.describe().T
        stats["missing"] = X.isnull().sum()
        stats["missing_pct"] = X.isnull().mean() * 100
        stats["skewness"] = X.skew()
        stats["kurtosis"] = X.kurtosis()
        return stats

    def save_splits(self, splits: Dict, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save train/val/test splits to disk.

        Args:
            splits: Dictionary of data splits
            output_dir: Directory to save files

        Returns:
            Dictionary of saved file paths
        """
        output_dir = output_dir or settings.PROCESSED_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for name, data in splits.items():
            if data is not None:
                path = output_dir / f"{name}.csv"
                if isinstance(data, pd.DataFrame):
                    data.to_csv(path, index=False)
                elif isinstance(data, pd.Series):
                    data.to_csv(path, index=False, header=["smiles"])
                saved_paths[name] = path
                logger.debug(f"Saved {name} to {path}")

        return saved_paths
