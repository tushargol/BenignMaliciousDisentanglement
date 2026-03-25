from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    Centralized paths.
    By default we point at the dataset folder currently in the workspace root: `02-Semiurban/`.
    If you later move it under `data/02-Semiurban/`, update these fields or add detection logic.
    """

    repo_root: Path
    scenario_root: Path

    ipal_train_events: Path
    ipal_test_events: Path
    ipal_train_initial_state: Path
    ipal_test_initial_state: Path

    raw_train_events: Path
    raw_test_events: Path

    outputs_dir: Path

    @staticmethod
    def auto() -> "Paths":
        repo_root = Path(__file__).resolve().parents[1]

        # Prefer `data/02-Semiurban` if present, else fallback to workspace root `02-Semiurban`.
        cand1 = repo_root / "data" / "02-Semiurban"
        cand2 = repo_root / "02-Semiurban"
        scenario_root = cand1 if cand1.exists() else cand2

        ipal_train = scenario_root / "ipal" / "train"
        ipal_test = scenario_root / "ipal" / "test"
        raw_train = scenario_root / "raw" / "train"
        raw_test = scenario_root / "raw" / "test"

        return Paths(
            repo_root=repo_root,
            scenario_root=scenario_root,
            ipal_train_events=ipal_train / "events.json",
            ipal_test_events=ipal_test / "events.json",
            ipal_train_initial_state=ipal_train / "initial_state.json",
            ipal_test_initial_state=ipal_test / "initial_state.json",
            raw_train_events=raw_train / "events.jsonl",
            raw_test_events=raw_test / "events.jsonl",
            outputs_dir=repo_root / "outputs",
        )

