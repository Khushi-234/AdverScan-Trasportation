"""
Main XAI Coordinator (XAIExplainer) for AdverScan explainability module.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from app.attack_engine.models import AttackResult
from app.explainability.comparison import compare_explanations
from app.explainability.explanation_result import ExplanationResult
from app.explainability.failure_analysis import analyze_failure
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.shap_explainer import SHAPExplainer


class XAIExplainer:
    """
    Main coordinator for XAI explainability in AdverScan.

    Coordinates prediction extraction, confidence calculation, technique-specific
    attribution generation (SHAP / LIME), clean vs. adversarial comparison, and
    failure mode analysis.
    """

    def __init__(self, default_technique: str = "shap"):
        """
        Initialize XAIExplainer.

        Args:
            default_technique: Default XAI technique identifier ("shap" or "lime").
        """
        self.default_technique = default_technique.lower()
        self.techniques = {
            "shap": SHAPExplainer(),
            "lime": LIMEExplainer(),
        }

    def _get_predictions_and_confidences(
        self, model: Any, inputs: Any
    ) -> Tuple[Any, float]:
        """
        Extract model predicted class label and confidence score.

        Args:
            model: PyTorch module, PyTorchAdapter, or callable model.
            inputs: Input tensor or array.

        Returns:
            Tuple of (predicted_class, confidence_score).
        """
        if isinstance(inputs, torch.Tensor):
            t_input = inputs
        elif isinstance(inputs, np.ndarray):
            t_input = torch.from_numpy(inputs).float()
        elif isinstance(inputs, (list, tuple)):
            t_input = torch.tensor(inputs).float()
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            outputs = model.predict(t_input)
        elif callable(model):
            model_eval = model.eval() if hasattr(model, "eval") else model
            with torch.no_grad():
                outputs = model_eval(t_input)
        else:
            raise TypeError("Model must be callable or possess a predict method.")

        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs)

        outputs = outputs.float()

        # Apply softmax if outputs look like raw logits (not already normalized probabilities)
        if outputs.ndim >= 2:
            probs = torch.softmax(outputs, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)

            if len(preds) == 1:
                return preds.item(), float(confidences.item())
            else:
                return preds.cpu().tolist(), float(confidences.mean().item())
        elif outputs.ndim == 1:
            probs = torch.softmax(outputs, dim=0)
            conf, pred = torch.max(probs, dim=0)
            return pred.item(), float(conf.item())
        else:
            return outputs.item(), 1.0

    def explain(
        self,
        model: Any,
        clean_input: Any,
        adversarial_input: Any,
        true_label: Optional[Any] = None,
        technique: Optional[str] = None,
        attack_name: str = "adversarial_attack",
        assessment_result: Optional[Any] = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Generate complete XAI explanation for clean vs. adversarial inputs.

        Args:
            model: Target ML model.
            clean_input: Clean baseline input sample/batch.
            adversarial_input: Adversarial input sample/batch.
            true_label: Optional ground truth label.
            technique: XAI technique name ("shap" or "lime"). Defaults to self.default_technique.
            attack_name: Identifier of the attack executed.
            assessment_result: Optional AssessmentResult from M5 for context enrichment.

        Returns:
            Structured ExplanationResult DTO.
        """
        tech_name = (technique or self.default_technique).lower()

        if tech_name not in self.techniques:
            raise ValueError(
                f"Unsupported XAI technique '{tech_name}'. Available options: {list(self.techniques.keys())}"
            )

        # 1. Get predictions and confidences
        clean_pred, clean_conf = self._get_predictions_and_confidences(model, clean_input)
        adv_pred, adv_conf = self._get_predictions_and_confidences(model, adversarial_input)

        # 2. Run XAI technique
        explainer = self.techniques[tech_name]
        clean_attr_result = explainer.explain(model, clean_input, target_class=clean_pred, **kwargs)
        adv_attr_result = explainer.explain(model, adversarial_input, target_class=adv_pred, **kwargs)

        clean_attr = clean_attr_result.get("attribution")
        adv_attr = adv_attr_result.get("attribution")

        attribution_data = {
            "technique": tech_name,
            "clean": clean_attr_result,
            "adversarial": adv_attr_result,
        }

        # 3. Perform clean vs. adversarial comparison
        comp_data = compare_explanations(
            clean_prediction=clean_pred,
            adversarial_prediction=adv_pred,
            clean_confidence=clean_conf,
            adversarial_confidence=adv_conf,
            clean_attribution=clean_attr,
            adv_attribution=adv_attr,
        )

        # 4. Perform failure analysis
        fail_data = analyze_failure(
            clean_prediction=clean_pred,
            adversarial_prediction=adv_pred,
            true_label=true_label,
        )

        # 5. Metadata
        meta: Dict[str, Any] = {
            "attack_name": attack_name,
            "technique": tech_name,
        }
        if assessment_result is not None:
            if hasattr(assessment_result, "to_dict"):
                meta["assessment_result"] = assessment_result.to_dict()
            elif isinstance(assessment_result, dict):
                meta["assessment_result"] = assessment_result

        # 6. Build ExplanationResult DTO
        return ExplanationResult(
            attack_name=attack_name,
            technique=tech_name,
            true_label=true_label,
            clean_prediction=clean_pred,
            adversarial_prediction=adv_pred,
            clean_confidence=clean_conf,
            adversarial_confidence=adv_conf,
            prediction_changed=comp_data["prediction_changed"],
            attack_caused_failure=fail_data["attack_caused_failure"],
            attribution=attribution_data,
            comparison=comp_data,
            failure_analysis=fail_data,
            metadata=meta,
        )

    def explain_attack_result(
        self,
        model: Any,
        attack_result: AttackResult,
        assessment_result: Optional[Any] = None,
        technique: Optional[str] = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Generate explanation directly from an M3 AttackResult contract.

        Args:
            model: Target ML model.
            attack_result: M3 AttackResult instance.
            assessment_result: Optional M5 AssessmentResult instance.
            technique: XAI technique name ("shap" or "lime").

        Returns:
            Structured ExplanationResult DTO.
        """
        clean_input = attack_result.original_inputs
        adv_input = attack_result.adversarial_examples  # or attack_result.adv_inputs
        labels = attack_result.labels
        attack_name = (
            attack_result.metadata.attack_name
            if hasattr(attack_result, "metadata") and hasattr(attack_result.metadata, "attack_name")
            else "unknown_attack"
        )

        if clean_input is None:
            raise ValueError("AttackResult must contain 'original_inputs' for XAI explainability.")

        return self.explain(
            model=model,
            clean_input=clean_input,
            adversarial_input=adv_input,
            true_label=labels,
            technique=technique,
            attack_name=attack_name,
            assessment_result=assessment_result,
            **kwargs,
        )
