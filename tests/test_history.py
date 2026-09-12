"""Focused shape and mode checks for the optional causal graph history path."""

import unittest
import torch

from tocu.history import graph_history_features, history_features


class GraphHistoryAdapterTests(unittest.TestCase):
    def test_graph_history_modes_preserve_expected_widths(self):
        flow = torch.rand(2, 12, 5) * 100.0
        adjacency = torch.rand(5, 5)
        adjacency.fill_diagonal_(0.0)

        self.assertEqual(tuple(history_features(flow).shape), (2, 5, 10))
        self.assertEqual(tuple(history_features(flow, adjacency, graph_mode="outgoing").shape), (2, 5, 21))
        self.assertEqual(tuple(history_features(flow, adjacency, graph_mode="symmetric").shape), (2, 5, 21))
        self.assertEqual(tuple(history_features(flow, adjacency, graph_mode="bidirectional").shape), (2, 5, 32))
        self.assertEqual(tuple(graph_history_features(flow, adjacency, mode="bidirectional").shape), (2, 5, 22))


    def test_graph_history_rejects_unknown_mode(self):
        flow = torch.rand(1, 12, 3)
        adjacency = torch.ones(3, 3).fill_diagonal_(0.0)
        with self.assertRaisesRegex(ValueError, "unsupported graph history mode"):
            graph_history_features(flow, adjacency, mode="future")


if __name__ == "__main__":
    unittest.main()
