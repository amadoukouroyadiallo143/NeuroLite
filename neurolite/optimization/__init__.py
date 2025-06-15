"""
Module d'optimisation pour NeuroLite.
"""
from .quantization import apply_dynamic_quantization, apply_static_quantization

__all__ = [
    "apply_dynamic_quantization",
    "apply_static_quantization"
] 