"""
Agent factory functions for medical coding domains.

This module provides standardized agent creation functions for medical
coding systems like ICD-10-CM, supporting all agent frameworks.
"""

from .icd10_domain import (
    # Base agents
    create_icd10_base_agent,
    create_icd10_random_agent, 
    create_icd10_scorer,
    
    # LLM agents
    create_icd10_llm_agent,
    
    # DSPy agents
    create_icd10_dspy_agent,
    create_icd10_configurable_agent,  # deprecated
    
    # Convenience functions
    create_icd10_agent_suite,
    
    # Legacy functions (deprecated)
    create_medical_agent,
    create_llm_medical_agent,
)

__all__ = [
    # Base agents
    "create_icd10_base_agent",
    "create_icd10_random_agent",
    "create_icd10_scorer",
    
    # LLM agents  
    "create_icd10_llm_agent",
    
    # DSPy agents
    "create_icd10_dspy_agent",
    "create_icd10_configurable_agent",  # deprecated
    
    # Convenience functions
    "create_icd10_agent_suite",
    
    # Legacy functions (deprecated)
    "create_medical_agent",
    "create_llm_medical_agent",
]