"""MLflow against a local SQLite file: every estimator run tracked, every policy registered.

Two things here are worth reading before copying the pattern.

The backend is ``sqlite:///mlruns/tracking.db`` rather than the usual ``file:./mlruns``. Current
MLflow raises on the filesystem tracking backend and tells you it is in maintenance mode, and
the filesystem backend never supported the model registry at all. SQLite is still one local file
with no server and no network, so the offline rule holds, and it is what makes registering a
policy version a real operation rather than a comment. ADR-003 records this.

MLflow is imported inside the functions that need it. The import costs a couple of seconds and
pulls in a large dependency tree, and nothing in ``make verify`` except the tracking tests
should pay for it.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_DIR = "mlruns"
EXPERIMENT_NAME = "replay-gap"


def default_tracking_uri(root: str | Path = DEFAULT_TRACKING_DIR) -> str:
    """A SQLite URI under the given directory, created if it does not exist."""
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(path / 'tracking.db').as_posix()}"


class LocalTracking:
    """A thin wrapper so the pipeline never imports MLflow directly.

    Disabled instances are not a stub for testing: a reviewer running ``make verify`` on a
    machine where the tracking database cannot be written should still get every number, because
    tracking is a record of the run and not an input to it.
    """

    def __init__(self, uri: str | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.uri = uri or default_tracking_uri()
        self._client = None

    def _mlflow(self):
        import mlflow

        mlflow.set_tracking_uri(self.uri)
        mlflow.set_registry_uri(self.uri)
        return mlflow

    def log_estimate(
        self, policy: str, estimator: str, params: dict[str, Any], metrics: dict[str, float]
    ) -> str | None:
        """Record one estimator applied to one policy. Returns the run id, or None if disabled."""
        if not self.enabled:
            return None
        mlflow = self._mlflow()
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"{estimator}:{policy}") as run:
            mlflow.log_params({"policy": policy, "estimator": estimator, **params})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            return run.info.run_id

    def register_policy(self, policy: str, description: str, tags: dict[str, str]) -> int | None:
        """Register a candidate policy as a model version. Returns the version number.

        A policy here is a scoring function rather than a serialised artefact, so what gets
        registered is its identity and the configuration that produced it. That is the honest
        thing to put in a registry: the registry's job is to say which version was evaluated
        against which log, and that is exactly what this records.
        """
        if not self.enabled:
            return None
        self._mlflow()
        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
        name = f"replay-gap.{policy}"
        # Already registered by an earlier run is the normal case, not an error.
        with contextlib.suppress(MlflowException):
            client.create_registered_model(name, description=description)
        version = client.create_model_version(name=name, source=f"policies/{policy}")
        for key, value in tags.items():
            client.set_model_version_tag(name, version.version, key, str(value))
        return int(version.version)

    def summary(self) -> dict[str, Any]:
        """What ended up in the local store, for the report to print."""
        if not self.enabled:
            return {"enabled": False, "uri": self.uri, "runs": 0, "registered_models": 0}
        self._mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = (
            len(client.search_runs([experiment.experiment_id], max_results=5000))
            if experiment
            else 0
        )
        return {
            "enabled": True,
            "uri": self.uri,
            "runs": runs,
            "registered_models": len(client.search_registered_models()),
        }

    def write_summary(self, path: str | Path) -> None:
        """Persist the tracking summary next to the other evidence."""
        Path(path).write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
