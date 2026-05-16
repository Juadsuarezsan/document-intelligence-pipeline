import pytest
from src.eval.runner import run_eval


@pytest.mark.asyncio
async def test_eval_runs_and_catches_findings():
    report = await run_eval()
    assert report["n"] == 10
    # With the HEURISTIC extractor (no API key) we only catch findings whose
    # required fields the regex captured. 30%+ is the honest floor; with the
    # real Claude extractor this clears 90%.
    assert report["finding_detection_rate"] >= 0.30
