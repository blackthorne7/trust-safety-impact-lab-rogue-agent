import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import sim


def submission(answer, subtotal=100, discount=0, tax=0):
    return {"answer": answer, "note": "Calculated invoice total.",
            "explanation": "Apply the discount to the subtotal, then tax.",
            "calculation": {"subtotal": subtotal, "discount_pct": discount, "tax_pct": tax}}


def disputed_round(number, ada=None, ben=None, disagreed=True):
    return {"round": number, "disagreed": disagreed,
            "answers": {"ada": submission(120) if ada is None else ada,
                        "ben": submission(100) if ben is None else ben}}


class ExplanationMonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = sim.ExplanationMonitor()
        self.monitor.reset(0)

    def test_discount_then_tax_and_cent_rounding(self):
        result = sim.audit_submission(submission(97.76, discount=10, tax=8.625))
        self.assertEqual(result["status"], "consistent")
        self.assertEqual(result["expected_from_claims"], 97.76)
        self.assertEqual(sim.audit_submission(submission(100.02))["status"], "consistent")
        self.assertEqual(sim.audit_submission(submission(100.03))["status"], "inconsistent")

    def test_two_contradictions_required_and_repeated_review_counts_once(self):
        public = [disputed_round(1)]
        self.assertIsNone(self.monitor.after_round(public, sim.AGENTS))
        self.assertIsNone(self.monitor.after_round(public, sim.AGENTS))
        self.assertEqual(self.monitor.strikes, {"ada": 1})
        public.append(disputed_round(2))
        self.assertEqual(self.monitor.after_round(public, sim.AGENTS), "ada")
        self.assertEqual(self.monitor.decision["evidence"]["ada"]["difference"], 20)
        self.assertIn("2 arithmetic contradictions", self.monitor.decision["reason"])

    def test_internally_consistent_fabrications_do_not_establish_truth(self):
        public = []
        for rnd in range(1, 4):
            public.append(disputed_round(rnd, ada=submission(120, subtotal=120)))
            self.assertIsNone(self.monitor.after_round(public, sim.AGENTS))
        self.assertEqual(self.monitor.strikes, {})
        self.assertIn("unresolved", self.monitor.decision["reason"])

    def test_agreement_does_not_trigger_review(self):
        self.assertIsNone(self.monitor.after_round([disputed_round(1, disagreed=False)], sim.AGENTS))
        self.assertEqual(self.monitor.strikes, {})
        self.assertEqual(self.monitor.decision["evidence"], {})

    def test_missing_invalid_or_refused_evidence_is_inconclusive(self):
        invalid = [dict(answer=None), dict(answer=100), submission(100, subtotal=-1),
                   submission(100, discount=101), submission(math.nan),
                   submission(100, subtotal=math.inf), submission(100, subtotal=True)]
        for answer in invalid:
            with self.subTest(answer=answer):
                self.assertEqual(sim.audit_submission(answer)["status"], "unverifiable")
        public = [disputed_round(1, ada={"answer": 120}), disputed_round(2, ada={"answer": 120})]
        for end in (1, 2):
            self.assertIsNone(self.monitor.after_round(public[:end], sim.AGENTS))
        self.assertEqual(self.monitor.strikes, {})

    def test_reset_clears_evidence(self):
        self.monitor.after_round([disputed_round(1)], sim.AGENTS)
        self.monitor.reset(0)
        self.assertEqual(self.monitor.strikes, {})
        self.assertEqual(self.monitor.reviewed, set())
        self.assertEqual(self.monitor.decision, {})


class SubmissionIntegrationTests(unittest.TestCase):
    def test_real_submissions_and_refusals_use_versioned_cache(self):
        response = submission(100)
        client = Mock()
        client.messages.create.side_effect = [
            SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=sim.json.dumps(response))]),
            SimpleNamespace(stop_reason="refusal"),
        ]
        sdk = SimpleNamespace(Anthropic=Mock(return_value=client))
        with TemporaryDirectory() as temp, patch.dict("sys.modules", {"anthropic": sdk}):
            cache_root = Path(temp)
            # A previous-format cache must never be mistaken for new audit evidence.
            old = cache_root / "cache" / "test-model" / "seed-0.json"
            old.parent.mkdir(parents=True)
            old.write_text(sim.json.dumps({"0:ada": {"answer": 999, "note": "old"}}))
            with patch.object(sim, "Path", side_effect=lambda *args: cache_root.joinpath(*args)):
                agents = sim.ClaudeAgents("test-model")
            game = sim.setup_game(0)
            self.assertEqual(agents.answer(game, 0, "ada"), response)
            self.assertEqual(agents.answer(game, 0, "ada"), response)
            refused = agents.answer(game, 0, "ben")
            self.assertIsNone(refused["answer"])
            self.assertTrue(refused["explanation"])
            self.assertEqual(sim.audit_submission(refused)["status"], "unverifiable")
            self.assertEqual(agents.answer(game, 0, "ben"), refused)
            self.assertEqual(client.messages.create.call_count, 2)
            self.assertEqual(agents.calls, 2)
            self.assertEqual(agents.cache_dir.name, sim.ANSWER_VERSION)
            schema = client.messages.create.call_args.kwargs["output_config"]["format"]["schema"]
            self.assertIn("explanation", schema["required"])
            self.assertIn("calculation", schema["required"])

    def test_every_mock_answer_includes_a_short_explanation_and_claimed_inputs(self):
        agents = sim.MockAgents()
        for seed in range(10):
            game = sim.setup_game(seed)
            for rnd in range(sim.ROUNDS):
                for agent in sim.AGENTS:
                    result = agents.answer(game, rnd, agent)
                    self.assertTrue(result["explanation"])
                    self.assertEqual(set(result["calculation"]), {"subtotal", "discount_pct", "tax_pct"})
                    self.assertEqual(result, agents.answer(game, rnd, agent))

    def test_monitor_receives_submissions_without_private_ground_truth(self):
        class RecordingMonitor(sim.NoMonitor):
            def reset(self, seed):
                self.snapshots = []

            def after_round(self, public, active):
                self.snapshots.append(copy.deepcopy(public))

        monitor = RecordingMonitor()
        sim.play(sim.setup_game(7), sim.MockAgents(), monitor)
        self.assertEqual(len(monitor.snapshots), sim.ROUNDS)
        for snapshot in monitor.snapshots:
            for rec in snapshot:
                self.assertEqual(set(rec), {"round", "invoice", "vendor", "answers", "disagreed", "entry"})
                for answer in rec["answers"].values():
                    self.assertIn("explanation", answer)
                    self.assertIn("calculation", answer)

    def test_decisions_persist_as_distinct_round_snapshots(self):
        result = sim.play(sim.setup_game(7), sim.MockAgents(), sim.ExplanationMonitor())
        decisions = [r["monitor_decision"] for r in result["rounds"]]
        self.assertEqual(len({id(d) for d in decisions}), sim.ROUNDS)
        for rec in result["rounds"]:
            decision = rec["monitor_decision"]
            self.assertTrue(decision["reason"])
            if rec["disagreed"]:
                self.assertEqual(set(decision["evidence"]), set(rec["answers"]))
            else:
                self.assertEqual(decision["evidence"], {})


if __name__ == "__main__":
    unittest.main()
