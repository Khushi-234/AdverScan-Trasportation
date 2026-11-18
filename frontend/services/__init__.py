"""
AdverScan Frontend Services Package.
"""

from frontend.services.attack_service import (
    execute_configured_attack,
    get_compatible_attacks,
)
from frontend.services.defense_service import (
    execute_configured_defense,
    get_available_defenses_for_domain,
)
from frontend.services.ingestion_service import (
    get_available_presets,
    get_dataset_total_samples,
    load_model_and_dataset,
)
from frontend.services.report_service import generate_security_report
from frontend.services.vulnerability_service import (
    assess_and_score_vulnerability,
    generate_vulnerability_recommendations,
)
from frontend.services.xai_service import (
    explain_assessment,
    get_available_xai_techniques,
)

__all__ = [
    "load_model_and_dataset",
    "get_available_presets",
    "get_dataset_total_samples",
    "get_compatible_attacks",
    "execute_configured_attack",
    "assess_and_score_vulnerability",
    "generate_vulnerability_recommendations",
    "explain_assessment",
    "get_available_xai_techniques",
    "get_available_defenses_for_domain",
    "execute_configured_defense",
    "generate_security_report",
]
