"""
Local feature store using Parquet files with versioning.

Provides a simple, versioned feature store that:
  - Stores feature DataFrames as Parquet files partitioned by version.
  - Loads features for a specific version or the latest.
  - Tracks metadata (creation time, feature names, row count).
  - Supports incremental appends.

For production scale, migrate to Feast or Tecton.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LocalFeatureStore:
    """
    Versioned local feature store backed by Parquet files.

    Directory layout::

        <root>/
        ├── v1/
        │   ├── features.parquet
        │   └── metadata.yaml
        ├── v2/
        │   ├── features.parquet
        │   └── metadata.yaml
        └── latest -> v2

    Args:
        root: Root directory for the feature store.
        auto_version: If True, auto-increment version on each write.
    """

    def __init__(
        self,
        root: Union[str, Path] = "data/features",
        auto_version: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.auto_version = auto_version

    # ── Version management ──────────────────────────────────────────

    def _current_version(self) -> int:
        """Return the latest version number (0 if none)."""
        versions = [
            int(d.name[1:])
            for d in self.root.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ]
        return max(versions) if versions else 0

    def _version_path(self, version: int) -> Path:
        return self.root / f"v{version}"

    def _latest_symlink(self) -> Path:
        return self.root / "latest"

    def next_version(self) -> int:
        """Return the next available version number."""
        if self.auto_version:
            return self._current_version() + 1
        return self._current_version()

    # ── Read / Write ────────────────────────────────────────────────

    def write(
        self,
        features: pd.DataFrame,
        version: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Write a feature DataFrame to the store.

        Args:
            features: Feature DataFrame with a ``DatetimeIndex``.
            version: Version number. Auto-increments if not given.
            metadata: Additional metadata to store.

        Returns:
            Version number used.
        """
        if version is None:
            version = self.next_version()

        vpath = self._version_path(version)
        vpath.mkdir(parents=True, exist_ok=True)

        # Write Parquet
        parquet_path = vpath / "features.parquet"
        features.to_parquet(parquet_path)

        # Write metadata
        import yaml as _yaml

        meta = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shape": list(features.shape),
            "columns": list(features.columns),
            "dtypes": {str(k): str(v) for k, v in features.dtypes.items()},
            "date_range": (
                str(features.index.min()),
                str(features.index.max()),
            ),
            "user_metadata": metadata or {},
        }
        meta_path = vpath / "metadata.yaml"
        with open(meta_path, "w") as f:
            _yaml.dump(meta, f, default_flow_style=False)

        # Update latest symlink
        self._update_latest(version)

        logger.info(
            "Feature store: wrote v%d (%d rows, %d cols)",
            version,
            features.shape[0],
            features.shape[1],
        )
        return version

    def read(
        self,
        version: Optional[int] = None,
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read features from the store.

        Args:
            version: Version to read. Reads ``latest`` if None.
            columns: Subset of columns to load.
            start_date: Filter index >= this date.
            end_date: Filter index <= this date.

        Returns:
            Feature DataFrame.
        """
        vpath = self._resolve_version(version)
        parquet_path = vpath / "features.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No features found at {parquet_path}"
            )

        df = pd.read_parquet(parquet_path)
        if columns:
            df = df[columns]
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        logger.debug("Feature store: read v%d (%d rows)", self._version_from_path(vpath), len(df))
        return df

    def read_metadata(self, version: Optional[int] = None) -> Dict:
        """Read metadata for a given version."""
        vpath = self._resolve_version(version)
        meta_path = vpath / "metadata.yaml"
        if not meta_path.exists():
            return {}
        import yaml as _yaml

        with open(meta_path) as f:
            return _yaml.safe_load(f)

    # ── Version listing ─────────────────────────────────────────────

    def list_versions(self) -> List[int]:
        """List all available versions (sorted ascending)."""
        versions = []
        for d in self.root.iterdir():
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
                versions.append(int(d.name[1:]))
        return sorted(versions)

    def get_feature_names(self, version: Optional[int] = None) -> List[str]:
        """Return column names for a given version."""
        df = self.read(version)
        return list(df.columns)

    # ── Internal helpers ────────────────────────────────────────────

    def _resolve_version(self, version: Optional[int] = None) -> Path:
        if version is not None:
            return self._version_path(version)
        latest = self._latest_symlink()
        if latest.exists() and latest.is_symlink():
            return latest.resolve()
        # Fall back to highest version
        v = self._current_version()
        if v == 0:
            raise FileNotFoundError("No versions exist in feature store.")
        return self._version_path(v)

    def _update_latest(self, version: int) -> None:
        latest = self._latest_symlink()
        target = self._version_path(version)
        if latest.exists():
            latest.unlink()
        latest.symlink_to(target, target_is_directory=True)

    @staticmethod
    def _version_from_path(path: Path) -> int:
        return int(path.name[1:])
