#!/usr/bin/env python3
"""Output-contract tests for the v2.1 audit serializer."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import uniform_peak_core as peak
import uniform_peak_tracking_v21 as tracking
import uniform_result_writer_v21 as writer


def _consensus(identifier: str, pressure: float, index: int, q: float) -> peak.PressureConsensus:
    return peak.PressureConsensus(
        consensus_id=identifier,
        channel="spots",
        pressure=pressure,
        pressure_index=index,
        q=q,
        fwhm_q=0.02,
        relative_area=0.1,
        support=6,
        total_scans=6,
        required_support=5,
        member_keys=(),
        reliable=True,
    )


class V21AuditWriterTests(unittest.TestCase):
    def test_required_audits_are_populated_without_alias_columns(self) -> None:
        nodes = tuple(
            peak.TrackNode(
                consensus_id=f"c{index}",
                pressure=float(index),
                pressure_index=index - 1,
                q=1.0 + 0.01 * index,
                fwhm_q=0.02,
                relative_area=0.1,
                support=6,
            )
            for index in (1, 2, 3)
        )
        segment = tracking.SegmentedTrack(
            track_id="radial_peak_001_segment_01",
            parent_track_id="radial_peak_001",
            channel="spots",
            nodes=nodes,
            official=True,
            ambiguous=False,
            minimum_pressure_support=3,
            segment_index=1,
            segment_count=1,
            status="official",
        )
        link = tracking.TrajectoryLinkEvidence(
            edge_id="trajectory_edge_000001",
            first_consensus_id="c1",
            second_consensus_id="c2",
            first_pressure_GPa=1.0,
            second_pressure_GPa=2.0,
            first_pressure_index=0,
            second_pressure_index=1,
            first_q_A_inv=1.01,
            second_q_A_inv=1.02,
            missing_pressure_levels=0,
            forward_evaluated=True,
            backward_evaluated=True,
            forward_admissible=True,
            backward_admissible=True,
            within_original_gate=True,
            forward_matched=True,
            backward_matched=True,
            forward_cost=0.1,
            backward_cost=0.1,
            forward_source_margin=0.5,
            forward_target_margin=0.5,
            backward_source_margin=0.5,
            backward_target_margin=0.5,
            mutual=True,
            endpoint_quarantined=False,
            crosses_quarantined_boundary=False,
            order_crossing=False,
            accepted=True,
            cut_reason="",
        )
        result = tracking.SegmentedTrackingResult(
            segments=(segment,),
            pass_links=(),
            competitions=(),
            link_evidence=(link,),
            ambiguity_events=(),
            gap_cuts=(
                tracking.GapCutEvidence(
                    direction="low_to_high",
                    source_consensus_id="c3",
                    source_pressure_index=2,
                    first_unreachable_pressure_index=6,
                    missing_pressure_levels=3,
                ),
            ),
            quarantined_node_ids=(),
            parents=(
                tracking.ParentTrack(
                    parent_track_id="radial_peak_001",
                    channel="spots",
                    node_ids=("c1", "c2", "c3"),
                    segment_ids=(segment.track_id,),
                    had_cuts=False,
                ),
            ),
        )
        consensus = {
            float(index): (_consensus(f"c{index}", float(index), index - 1, 1.0 + 0.01 * index),)
            for index in (1, 2, 3)
        }
        analysis = SimpleNamespace(consensus_by_pressure=consensus, tracks=(segment,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = writer._write_v21_audits(root, "spots", analysis, result)
            for name in (
                "link_evidence.csv",
                "ambiguity_events.csv",
                "segment_lineage.csv",
                "quarantined_nodes.csv",
                "selection_audit.csv",
                "selection_audit_summary.csv",
                "gap_cuts.csv",
            ):
                self.assertTrue((root / name).is_file(), name)
            with (root / "link_evidence.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["first_pressure_GPa"], "1")
            self.assertEqual(rows[0]["first_q_A^-1"], "1.01")
            self.assertNotIn("first_q_A_inv", rows[0])
            self.assertEqual(rows[0]["forward_source_margin"], "0.5")
            with (root / "selection_audit_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                selection = list(csv.DictReader(handle))
            overall = next(row for row in selection if row["row_kind"] == "overall_retention")
            self.assertEqual(overall["all_reliable_nodes"], "3")
            self.assertEqual(overall["official_retained_nodes"], "3")
            self.assertEqual(overall["retained_fraction"], "1")
            self.assertEqual(metrics["accepted_links"], 1)
            self.assertEqual(metrics["missing_too_long_gap_cuts"], 1)


if __name__ == "__main__":
    unittest.main()
