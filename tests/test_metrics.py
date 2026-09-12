from __future__ import annotations

import unittest

import torch

from tocu.head import metric_totals


class MetricTests(unittest.TestCase):
    def test_seattle_mape_uses_high_flow_mask(self) -> None:
        target = torch.tensor([[[10.0, 20.0]]])
        samples = torch.tensor([[[[0.0, 10.0]], [[0.0, 10.0]]]])
        totals = metric_totals(samples, target, "Seattle")
        self.assertEqual(totals["MAPE_COUNT"], 1.0)


if __name__ == "__main__":
    unittest.main()
