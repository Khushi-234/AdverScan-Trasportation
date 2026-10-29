"""
Execution script to perform clean Baseline Evaluation & XAI Explanation Pipeline in AdverScan.
Supports planning (--plan), technique overrides (--technique), mode selection (--xai-mode),
budget control (--budget), and skipping (--skip-xai).
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.explainability.config import XAIConfig
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.modality import Modality
from app.explainability.techniques.registry import select_by_mode, technique_registry


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for baseline evaluation and XAI pipeline."""
    parser = argparse.ArgumentParser(
        description="AdverScan — Baseline Evaluation and Automatic XAI Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="Target model identifier or path (default: cardiffnlp/twitter-roberta-base-sentiment-latest)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name or split",
    )
    parser.add_argument(
        "--xai-mode",
        type=str,
        default="standard",
        choices=["fast", "standard", "thorough"],
        help="XAI evaluation mode preset ('fast' | 'standard' | 'thorough') (default: standard)",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default=None,
        help="Optional manual override for XAI technique (e.g. layer_integrated_gradients, saliency, shap)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Maximum number of samples to explain (default: 100)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print [XAI PLAN] and exit immediately without executing evaluation or inference",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip the entire XAI stage",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-sample timeout in seconds for timeout fallback (default: 30)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Execute rapid standalone demonstration in terminal",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda')",
    )
    return parser.parse_args(args)


class DemoTransformerTextModel(nn.Module):
    """Lightweight demonstration text model with embedding layer and classifier."""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, batch_first=True, dim_feedforward=64),
            num_layers=1,
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        emb = self.embeddings(x)
        h = self.encoder(emb)
        pooled = h.mean(dim=1)
        return self.fc(pooled)


from app.explainability.pipeline import (
    DemoTransformerTextModel,
    XAIPipeline,
    display_xai_plan,
)


def run_xai_pipeline(
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Execute the XAI pipeline with CLI configuration, terminal messages, and fallback handling."""
    cfg = XAIConfig(
        mode=args.xai_mode,
        technique=args.technique,
        budget=args.budget,
        per_sample_timeout_s=args.timeout,
    )
    pipeline = XAIPipeline(config=cfg)

    return pipeline.run(
        model_name=args.model,
        mode=args.xai_mode,
        technique_override=args.technique,
        budget=args.budget,
        device=args.device,
        timeout=args.timeout,
        plan_only=args.plan,
        skip_xai=args.skip_xai,
    )


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    run_xai_pipeline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
