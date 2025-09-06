"""
ICD-10-CM coding agent factories for all frameworks.

This module provides agent creation functions specifically configured for
ICD-10-CM diagnostic coding across all agent frameworks (base, LLM, DSPy).
"""

from typing import Any

from core.dag_agents.base_agents import DeterministicAgent, RandomAgent, SimpleScorer
from core.dag_agents.llm_agents import LLMAgent, create_llm_agent
from core.dag_agents.dspy_agents import ConfigurableDSPyAgent, create_configurable_agent, ICD10CodingSignature


# =============================================================================
# Base Agents (Manual/Deterministic Control)
# =============================================================================

def create_icd10_base_agent() -> DeterministicAgent:
    """Create manual/deterministic ICD-10 agent for testing and validation."""
    return DeterministicAgent()


def create_icd10_random_agent(seed: int | None = None) -> RandomAgent:
    """Create random ICD-10 agent for testing and baseline comparison."""
    return RandomAgent(seed=seed)


def create_icd10_scorer() -> SimpleScorer:
    """Create ICD-10 candidate scorer for scoring-only workflows."""
    return SimpleScorer()


# =============================================================================
# LLM Agents (Direct API Automation)
# =============================================================================

def create_icd10_llm_agent(provider: str = "openai", model: str = "gpt-4o") -> LLMAgent:
    """Create LLM-powered ICD-10-CM diagnostic coding agent."""
    return create_llm_agent(provider=provider, model=model, domain="icd10")


# =============================================================================
# DSPy Agents (Self-Optimizing Structured Output)
# =============================================================================

def create_icd10_dspy_agent(
    provider: str = "openai", 
    model: str = "gpt-4o",
    max_candidates: int = 1,
    temperature: float | None = None,
    reasoning_style: str | None = None,
    custom_prompts: dict[str, str] | None = None  # Keep for backward compatibility
) -> ConfigurableDSPyAgent:
    """
    Create DSPy-powered ICD-10-CM diagnostic coding agent with self-optimization.
    
    Args:
        provider: LLM provider ("openai", "anthropic", "cerebras", etc.)
        model: Model name ("gpt-4o", "claude-sonnet-4", "qwen-3-32b", etc.)
        max_candidates: Maximum candidates to select (for beam vs greedy search)
        temperature: Sampling temperature (None = provider default)
        reasoning_style: Reasoning approach - affects how DSPy agent reasons through decisions:
            - "detailed_clinical": Comprehensive reasoning with differential diagnoses 
            - "concise": Focused, brief reasoning for primary decision
            - "structured": Systematic reasoning with clear criteria
            - "evidence_based": Citation-focused reasoning from source documents
            - "differential": Comparative reasoning across diagnostic options
            - Custom string: Will be used as reasoning guidance directly
        custom_prompts: Advanced custom prompt overrides (backward compatibility)
        
    Returns:
        Configured DSPy agent ready for ICD-10 coding with add_rule() optimization
        
    Examples:
        # Simple usage
        agent = create_icd10_dspy_agent("openai", "gpt-4o")
        
        # Beam search with custom reasoning
        agent = create_icd10_dspy_agent("cerebras", "qwen-3-32b", max_candidates=3, 
                                      reasoning_style="detailed_clinical")
        
        # Add node-specific training after creation
        agent.add_rule("E11.22", training_examples_for_diabetes_complications)
    """
    try:
        from core.llm import get_model_config
        config = get_model_config(provider, model)
        
        # Apply temperature override if provided
        if temperature is not None:
            config = config.copy()
            config["temperature"] = temperature
            
        # Ensure high max_tokens for reasoning models (minimum 128000 for complex reasoning)
        config = config.copy()
        current_max_tokens = config.get("max_tokens", 8000)
        if current_max_tokens < 128000:
            config["max_tokens"] = 128000
            
        # Build prompts from intuitive parameters
        prompts = custom_prompts or {}
        if reasoning_style:
            if reasoning_style == "detailed_clinical":
                prompts["reasoning_instructions"] = "Provide detailed clinical reasoning with differential diagnosis considerations, comorbidity analysis, and comprehensive clinical context evaluation"
            elif reasoning_style == "concise":
                prompts["reasoning_instructions"] = "Provide concise, focused clinical reasoning that directly supports the primary diagnostic decision"
            elif reasoning_style == "structured":
                prompts["reasoning_instructions"] = "Use structured clinical decision-making with clear criteria, systematic evaluation, and evidence-based reasoning steps"
            elif reasoning_style == "evidence_based":
                prompts["reasoning_instructions"] = "Focus on evidence-based clinical reasoning with clear citations from the source document supporting each diagnostic decision"
            elif reasoning_style == "differential":
                prompts["reasoning_instructions"] = "Emphasize differential diagnosis reasoning, comparing and contrasting potential diagnostic options with clinical justification"
            else:
                # Custom reasoning style - use as-is
                prompts["reasoning_instructions"] = f"Apply {reasoning_style} reasoning approach to clinical decision-making"
        
        return ConfigurableDSPyAgent(
            signature_class=ICD10CodingSignature,
            domain_name="icd10",
            model_config=config,
            custom_prompts=prompts,
            max_candidates=max_candidates
        )
    except Exception as e:
        print(f"Warning: Could not configure {provider}/{model}: {e}")
        # Fallback config with high max_tokens for reasoning
        fallback_config = {"max_tokens": 128000, "temperature": 0.7}
        return ConfigurableDSPyAgent(
            signature_class=ICD10CodingSignature, 
            domain_name="icd10", 
            model_config=fallback_config,
            custom_prompts=custom_prompts or {},
            max_candidates=max_candidates
        )


def create_icd10_configurable_agent(
    provider: str = "openai", 
    model: str = "gpt-4o",
    custom_prompts: dict[str, str] | None = None
) -> ConfigurableDSPyAgent:
    """
    Create configurable ICD-10 DSPy agent with custom prompts.
    
    DEPRECATED: Use create_icd10_dspy_agent() with intuitive parameters instead.
    
    Old: create_icd10_configurable_agent("openai", "gpt-4o", {"style": "detailed"})  
    New: create_icd10_dspy_agent("openai", "gpt-4o", reasoning_style="detailed_clinical")
    """
    import warnings
    warnings.warn(
        "create_icd10_configurable_agent is deprecated. Use create_icd10_dspy_agent() with intuitive parameters instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_icd10_dspy_agent(provider=provider, model=model, custom_prompts=custom_prompts)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_icd10_agent_suite(provider: str = "openai", model: str = "gpt-4o") -> dict[str, Any]:
    """Create complete suite of ICD-10 agents across all frameworks."""
    return {
        "base": create_icd10_base_agent(),
        "random": create_icd10_random_agent(),
        "scorer": create_icd10_scorer(),
        "llm": create_icd10_llm_agent(provider, model),
        "dspy": create_icd10_dspy_agent(provider, model)
    }


# =============================================================================
# Backward Compatibility (Legacy Function Names)
# =============================================================================

def create_medical_agent(provider: str = "openai", model: str = "gpt-4o") -> ConfigurableDSPyAgent:
    """
    Create DSPy agent for medical/ICD coding.
    
    DEPRECATED: Use create_icd10_dspy_agent(provider, model) instead.
    This function will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "create_medical_agent is deprecated. Use create_icd10_dspy_agent(provider, model) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_icd10_dspy_agent(provider, model)


def create_llm_medical_agent(provider: str = "openai", model: str = "gpt-4o") -> LLMAgent:
    """
    Create LLM agent for medical coding.
    
    DEPRECATED: Use create_icd10_llm_agent(provider, model) instead.  
    This function will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "create_llm_medical_agent is deprecated. Use create_icd10_llm_agent(provider, model) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_icd10_llm_agent(provider, model)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("ICD-10-CM Domain Agent Factories")
    print("=" * 50)
    
    try:
        # Create agents using different frameworks
        base_agent = create_icd10_base_agent()
        print(f"✅ Created ICD-10 base agent: {type(base_agent).__name__}")
        
        llm_agent = create_icd10_llm_agent("openai", "gpt-4o")
        print(f"✅ Created ICD-10 LLM agent: {llm_agent.provider}/{llm_agent.model}")
        
        dspy_agent = create_icd10_dspy_agent("anthropic", "claude-sonnet-4")
        print(f"✅ Created ICD-10 DSPy agent: {dspy_agent.domain_name}")
        
        # Legacy functions still work
        legacy_agent = create_medical_agent("openai", "gpt-4o")
        print(f"✅ Created medical agent (legacy): {legacy_agent.domain_name}")
        
        print(f"\n🎯 ICD-10-CM Domain Benefits:")
        print(f"• All agent frameworks available for ICD-10 diagnostic coding")
        print(f"• Progressive automation: manual → LLM → DSPy self-optimization")
        print(f"• Domain-specific configuration with backward compatibility")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()