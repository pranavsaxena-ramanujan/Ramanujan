import unittest

from ramanujan_shards import AdapterGraph, StageDefinition, plan_shards


class ShardPlannerTest(unittest.TestCase):
    def test_plans_branching_non_transformer_graph(self):
        graph = AdapterGraph(
            architecture_id="spectrogram-autoencoder",
            architecture_version="test",
            stages=[
                StageDefinition("preprocess", 10, entrypoints=["forward"]),
                StageDefinition("frequency-encoder", 45, ["preprocess"]),
                StageDefinition("time-encoder", 35, ["preprocess"]),
                StageDefinition("merge", 20, ["frequency-encoder", "time-encoder"]),
            ],
        )

        plans = plan_shards(graph, requested_shards=2)

        self.assertEqual(2, len(plans))
        stage_ids = [stage_id for plan in plans for stage_id in plan.stage_ids]
        self.assertEqual(
            ["preprocess", "frequency-encoder", "time-encoder", "merge"],
            stage_ids,
        )
        self.assertEqual(len(stage_ids), len(set(stage_ids)))

    def test_balances_uneven_stage_sizes_without_splitting_stages(self):
        graph = AdapterGraph(
            architecture_id="generic",
            architecture_version="test",
            stages=[
                StageDefinition("a", 80),
                StageDefinition("b", 10, ["a"]),
                StageDefinition("c", 10, ["b"]),
            ],
        )

        plans = plan_shards(graph, requested_shards=2)

        self.assertEqual(["a"], plans[0].stage_ids)
        self.assertEqual(["b", "c"], plans[1].stage_ids)
        self.assertEqual(80, plans[0].estimated_resident_bytes)
        self.assertEqual(20, plans[1].estimated_resident_bytes)

    def test_rejects_unknown_dependencies_and_cycles(self):
        unknown = AdapterGraph("unknown", "test", [StageDefinition("a", 1, ["missing"])])
        with self.assertRaisesRegex(ValueError, "unknown stage"):
            plan_shards(unknown, 1)

        cyclic = AdapterGraph(
            "cyclic",
            "test",
            [StageDefinition("a", 1, ["b"]), StageDefinition("b", 1, ["a"])],
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            plan_shards(cyclic, 1)


if __name__ == "__main__":
    unittest.main()