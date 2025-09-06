"""
Medical Domain Module
=====================

This module contains domain-specific implementations for medical coding systems:
- ICD-10-CM: International Classification of Diseases
- CPT: Current Procedural Terminology  
- SNOMED CT: Systematized Nomenclature of Medicine Clinical Terms

It also includes tree navigation and traversal engines specific to medical domains.
"""

# Medical domain implementations
from .factories.icd10_domain import *
# from .cpt_domain import *      # TODO: Implement CPT domain
# from .snomed_domain import *   # TODO: Implement SNOMED domain

# Tree navigation and traversal for medical domains
from .trees import *
from .traversals import *

__all__ = [
    # ICD-10 domain exports
    'create_icd10_agent',
    'create_icd10_dspy_agent', 
    'create_icd10_llm_agent',
    'ICD10CodingSignature',
    
    # CPT domain exports
    'create_cpt_agent',
    'create_cpt_dspy_agent',
    'create_cpt_llm_agent', 
    'CPTCodingSignature',
    
    # SNOMED domain exports
    'create_snomed_agent',
    'create_snomed_dspy_agent',
    'create_snomed_llm_agent',
    'SNOMEDCodingSignature',
    
    # Tree and traversal exports
    'create_navigator',
    'create_icd_traversal_engine',
]