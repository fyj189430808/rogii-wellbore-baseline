from __future__ import annotations

import json
from pathlib import Path

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS


def test_candidate_generator_output_order_matches_frozen_columns() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "f05a_candidate_generator_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["output_feature_columns"] == DIRECT_CANDIDATE_COLUMNS
