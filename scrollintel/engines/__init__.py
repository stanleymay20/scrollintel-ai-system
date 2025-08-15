"""
ScrollIntel v2.0+ Intelligence Engines
Advanced processing engines for AI, ML, and data operations.
"""

from .file_processor import FileProcessorEngine
from .base_engine import BaseEngine
from .automodel_engine import AutoModelEngine
from .scroll_viz_engine import ScrollVizEngine
from .scroll_forecast_engine import ScrollForecastEngine
from .scroll_model_factory import ScrollModelFactory
from .scroll_insight_radar import ScrollInsightRadar
from .insight_generator import InsightGenerator
from .roi_calculator import ROICalculator
from .consciousness_engine import ConsciousnessEngine, AwarenessEngine
from .intentionality_engine import IntentionalityEngine
from .intuitive_reasoning_engine import IntuitiveReasoning
from .personality_engine import PersonalityEngine
# from .style_transfer_engine import StyleTransferEngine, StyleType

# TODO: Add other engines when needed for testing
# from .organizational_resilience_engine import OrganizationalResilienceEngine
# from .xai_engine import ExplainXEngine
# from .multimodal_engine import MultimodalEngine
# from .vault_engine import ScrollVaultEngine
# from .cognitive_core import CognitiveCore
# from .billing_engine import ScrollBillingEngine

__all__ = [
    "FileProcessorEngine",
    "BaseEngine",
    "AutoModelEngine",
    "ScrollVizEngine",
    "ScrollForecastEngine",
    "ScrollModelFactory",
    "ScrollInsightRadar",
    "InsightGenerator",
    "ROICalculator",
    "ConsciousnessEngine",
    "AwarenessEngine",
    "IntentionalityEngine",
    "IntuitiveReasoning",
    "PersonalityEngine",
    "StyleTransferEngine",
    "StyleType",
    "StyleTransferRequest",
    "StyleTransferResult"
    # "OrganizationalResilienceEngine"
]