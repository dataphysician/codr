"""
DAG Agent implementations for candidate selection and scoring.

This package contains various agent implementations that can select
and rank traversal candidates using different strategies (LLM, DSPy, heuristic).
These agents work with DAG (Directed Acyclic Graph) traversal patterns.

Note: Domain-specific factory functions have been moved to the domains/ subfolder
for better organization. Import them directly:
    from core.domains.medical import icd10_domain, cpt_domain, snomed_domain
"""

# Core agent implementations
from .base_agents import CandidateAgent, CandidateScorer, DeterministicAgent, RandomAgent, SimpleScorer
from .llm_agents import LLMAgent, create_llm_agent
from .dspy_agents import (
    ConfigurableDSPyAgent, create_configurable_agent,
    GenericCodingSignature, ICD10CodingSignature, CPTCodingSignature, SNOMEDCodingSignature
)

# Note: Domain-specific factory functions are available in core.domains.medical
# Import them directly to avoid circular dependencies:
#   from core.domains.medical.icd10_domain import create_icd10_llm_agent, create_icd10_dspy_agent
#   from core.domains.medical.cpt_domain import create_cpt_llm_agent, create_cpt_dspy_agent  
#   from core.domains.medical.snomed_domain import create_snomed_llm_agent, create_snomed_dspy_agent

__all__ = [
    # Core interfaces
    "CandidateAgent",
    "CandidateScorer",
    
    # Base agents
    "DeterministicAgent",
    "RandomAgent", 
    "SimpleScorer",
    
    # LLM agents
    "LLMAgent",
    "create_llm_agent",
    
    # DSPy agents
    "ConfigurableDSPyAgent",
    "create_configurable_agent",
    "GenericCodingSignature",
    "ICD10CodingSignature",
    "CPTCodingSignature", 
    "SNOMEDCodingSignature",
]