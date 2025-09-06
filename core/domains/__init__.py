"""
Domain-specific implementations for different coding systems.

This module provides domain-specific implementations organized by
coding domain, each containing trees, traversals, and agent factories.
"""

# Import medical domain for easy access
from . import medical

__all__ = [
    "medical"
]