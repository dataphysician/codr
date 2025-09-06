"""
DSPy Agents for Multi-Domain Coding Systems
=============================================

This module provides DSPy agents for different coding domains (medical, procedural, terminology)
with configurable signatures and LLM providers. Configuration concerns are handled by core modules.
"""

import dspy
from typing import Any

from core import NodeId, Action, DecisionContext, RichCandidate, CandidateAgent


# =================================================================================
# Generic DSPy Base Signatures for Any Coding System
# =================================================================================

class GenericCodingSignature(dspy.Signature):
    """Base signature for selecting codes from any hierarchical coding system."""
    
    source_document: str = dspy.InputField(desc="Source document to analyze")
    current_position: str = dspy.InputField(desc="Current position in coding tree") 
    position_ancestry: list[str] = dspy.InputField(desc="Path from root to current position")
    candidate_codes: list[str] = dspy.InputField(desc="Available codes with descriptions")
    
    selected_codes: list[str] = dspy.OutputField(desc="Best codes ranked by relevance")
    citations: list[str] = dspy.OutputField(desc="Relevant text snippets from source document")
    reasoning_quality: str = dspy.OutputField(desc="Self-assess the quality of your reasoning trace and alignment with the clinical request. Rate as: EXCELLENT (highly confident in accuracy and thoroughness), GOOD (confident with minor gaps), MODERATE (adequate but some uncertainty), or POOR (low confidence or significant gaps)")


# =================================================================================
# Domain-Specific Coding Signatures
# =================================================================================

class ICD10CodingSignature(GenericCodingSignature):
    """Select best ICD-10-CM codes from candidates based on clinical context."""
    
    # Override field descriptions for ICD-10 medical context
    source_document: str = dspy.InputField(desc="Clinical documentation (notes, reports, diagnoses)")
    current_position: str = dspy.InputField(desc="Current position in ICD-10-CM tree") 
    position_ancestry: list[str] = dspy.InputField(desc="ICD-10-CM hierarchy path from root to current")
    candidate_codes: list[str] = dspy.InputField(desc="Available ICD-10-CM codes with medical descriptions")
    
    selected_codes: list[str] = dspy.OutputField(desc="All clinically relevant ICD-10-CM codes from the candidate list, ranked by clinical relevance. Include multiple codes when they represent different aspects or complications of the clinical scenario.")
    citations: list[str] = dspy.OutputField(desc="Clinical text supporting ICD-10-CM code selection")


class CPTCodingSignature(GenericCodingSignature):
    """Select best CPT codes from candidates based on procedure context."""
    
    # Override field descriptions for CPT procedural context
    source_document: str = dspy.InputField(desc="Procedure notes, operative reports, billing documentation")
    current_position: str = dspy.InputField(desc="Current position in CPT code tree")
    position_ancestry: list[str] = dspy.InputField(desc="CPT hierarchy path from root to current")
    candidate_codes: list[str] = dspy.InputField(desc="Available CPT procedure codes with descriptions")
    
    selected_codes: list[str] = dspy.OutputField(desc="Best CPT codes ranked by procedure match")
    citations: list[str] = dspy.OutputField(desc="Procedure text supporting CPT code selection")


class SNOMEDCodingSignature(GenericCodingSignature):
    """Select best SNOMED-CT codes from candidates based on clinical terminology."""
    
    # Override field descriptions for SNOMED-CT terminology context  
    source_document: str = dspy.InputField(desc="Clinical text with medical concepts and terminology")
    current_position: str = dspy.InputField(desc="Current position in SNOMED-CT concept tree")
    position_ancestry: list[str] = dspy.InputField(desc="SNOMED-CT hierarchy path from root concept")
    candidate_codes: list[str] = dspy.InputField(desc="Available SNOMED-CT concepts with definitions")
    
    selected_codes: list[str] = dspy.OutputField(desc="Best SNOMED-CT concepts ranked by semantic match")
    citations: list[str] = dspy.OutputField(desc="Clinical text supporting SNOMED-CT concept selection")


# =================================================================================
# Configurable DSPy Agent for Any Coding System
# =================================================================================

class ConfigurableDSPyAgent(CandidateAgent):
    """
    Configurable DSPy agent that works with any coding system and signature.
    
    This agent can be configured for different domains (medical, procedural, terminology)
    and different LLM providers while maintaining the same CandidateAgent interface.
    """
    
    def __init__(
        self, 
        signature_class: type[dspy.Signature],
        domain_name: str,
        model_config: dict[str, Any] | None = None,
        custom_prompts: dict[str, str] | None = None,
        max_candidates: int | None = None
    ):
        """
        Initialize configurable DSPy agent.
        
        Args:
            signature_class: DSPy signature to use (ICD10CodingSignature, CPTCodingSignature, etc.)
            domain_name: Domain identifier for logging/tracking
            model_config: Optional DSPy model configuration
            custom_prompts: Optional custom prompts for domain-specific behavior
            max_candidates: Maximum candidates to select (int).
                           None defaults to 1 candidate.
        """
        self.signature_class = signature_class
        self.domain_name = domain_name
        self.custom_prompts = custom_prompts or {}
        
        # Configure candidate selection count
        if max_candidates is None:
            self.max_candidates = 1  # Default to single candidate
        else:
            self.max_candidates = max(1, int(max_candidates))  # Ensure at least 1
        
        # Store DSPy model configuration for use in contexts
        self.lm = None
        if model_config:
            # Use our create_dspy_lm function to handle Qwen models properly
            try:
                from core.engines.dspy_engine import create_dspy_lm
                provider = model_config.get("provider", "openai")
                model_name = model_config.get("model", "gpt-4o")
                # Extract the model name without provider prefix
                if "/" in model_name:
                    model_name = model_name.split("/")[1]
                # Pass the modified config values (especially max_tokens) to our creation function
                config_kwargs = {k: v for k, v in model_config.items() if k not in ["provider", "model"]}
                self.lm = create_dspy_lm(provider, model_name, **config_kwargs)
                # Try to configure - if in async context, this will fail gracefully
                try:
                    dspy.configure(lm=self.lm)
                except RuntimeError:
                    # In async context - will use dspy.context() later
                    pass
            except Exception:
                # Fallback to direct creation if our function fails
                try:
                    self.lm = dspy.LM(**model_config)
                    dspy.configure(lm=self.lm)
                except RuntimeError:
                    # In async context - will use dspy.context() later  
                    pass
        
        # Initialize ChainOfThought module with the provided signature
        # No need to modify signature - truncation happens after DSPy returns results
        self.coder = dspy.ChainOfThought(signature_class)
        
        # Track node-specific signature customizations (like LLM agent rules)
        self.node_rules: dict[str, dict[str, str]] = {}  # {node_id: {field: custom_description}}
        self.node_coders: dict[str, dspy.ChainOfThought] = {}  # Cache of customized coders
        
        # Track training examples for future DSPy compilation  
        self.training_examples: list[dict[str, Any]] = []
    
    def _get_node_specific_coder(self, node_id: str) -> dspy.ChainOfThought:
        """Get DSPy coder with node-specific signature customizations (if any)."""
        # Check if we have node-specific rules
        if node_id not in self.node_rules:
            return self.coder  # Use default coder
            
        # Check cache first
        if node_id in self.node_coders:
            return self.node_coders[node_id]
            
        # Create customized signature class dynamically
        rules = self.node_rules[node_id]
        
        # Create a new signature class with customized field descriptions
        class CustomizedSignature(self.signature_class):
            """Dynamically customized signature with node-specific field descriptions."""
            pass
        
        # Apply field description customizations
        for field_name, custom_desc in rules.items():
            if hasattr(self.signature_class, field_name):
                original_field = getattr(self.signature_class, field_name)
                if hasattr(original_field, 'desc'):
                    # Create new field with custom description
                    if hasattr(original_field, '__class__') and 'InputField' in str(original_field.__class__):
                        setattr(CustomizedSignature, field_name, dspy.InputField(desc=custom_desc))
                    elif hasattr(original_field, '__class__') and 'OutputField' in str(original_field.__class__):
                        setattr(CustomizedSignature, field_name, dspy.OutputField(desc=custom_desc))
        
        # Create and cache the customized coder
        customized_coder = dspy.ChainOfThought(CustomizedSignature)
        self.node_coders[node_id] = customized_coder
        return customized_coder
    
    def _generate_optimized_field_description(
        self, 
        node_id: str, 
        field_name: str, 
        feedback: str, 
        model: str | None = None
    ) -> str:
        """
        Generate optimized field description using metaprompt optimization.
        
        This mirrors the LLM agent's metaprompt approach but optimizes DSPy signature field descriptions.
        """
        # Get original field description
        original_field = getattr(self.signature_class, field_name, None)
        original_desc = original_field.desc if original_field and hasattr(original_field, 'desc') else f"Field description for {field_name}"
        
        # Determine optimization model
        optimization_model = model or f"openai/gpt-4o"  # Default model for optimization
        
        # Build metaprompt for DSPy field description optimization
        metaprompt = f"""
DSPy SIGNATURE FIELD OPTIMIZATION REQUEST

TASK: Optimize a DSPy signature field description to incorporate user feedback and improve reasoning guidance.

CONTEXT:
- Domain: ICD-10-CM medical coding
- Target Node: {node_id}
- Field Name: {field_name}
- Signature Class: {self.signature_class.__name__}

CURRENT FIELD DESCRIPTION:
{original_desc}

USER FEEDBACK:
{feedback}

OPTIMIZATION REQUIREMENTS:
1. Incorporate the user feedback to guide the LLM's reasoning process
2. Maintain the technical purpose of the field
3. Be specific enough to influence model behavior
4. Keep the description concise but comprehensive
5. Focus on the specific node context: {node_id}

Generate an optimized field description that will guide the DSPy ChainOfThought module to apply the feedback when processing node {node_id}.

OPTIMIZED FIELD DESCRIPTION:
"""
        
        try:
            # Import LLM call function
            from core.engines.llm_engine import call_llm
            
            # Make optimization call
            messages = [
                {"role": "system", "content": "You are a DSPy signature optimization expert specializing in medical coding field descriptions."},
                {"role": "user", "content": metaprompt}
            ]
            
            # Parse provider/model
            if "/" in optimization_model:
                opt_provider, opt_model = optimization_model.split("/", 1)
            else:
                opt_provider, opt_model = "openai", optimization_model
            
            response = call_llm(
                messages=messages,
                provider=opt_provider,
                model=opt_model,
                max_tokens=200,  # Field descriptions should be concise
                temperature=0.2  # Lower temperature for consistent optimization
            )
            
            return response.strip()
            
        except Exception as e:
            # Fallback to simple enhancement if metaprompt fails
            print(f"⚠️  Metaprompt optimization failed for {node_id}.{field_name}: {e}")
            if field_name == "selected_codes":
                return f"Best ICD-10-CM codes selected using: {feedback}. Rank by clinical relevance with systematic reasoning."
            elif field_name == "reasoning_quality":
                return f"Self-assess reasoning quality. Apply {feedback} and rate as EXCELLENT, GOOD, MODERATE, or POOR."
            else:
                return f"{original_desc} Apply guidance: {feedback}"
        
    def select(self, decision_context: DecisionContext, candidates: list[tuple[NodeId, Action]]) -> list[RichCandidate]:
        """Select best candidates using configured DSPy signature."""
        if not candidates:
            return []
        
        # Extract current node information
        current_node = decision_context.node
        if not current_node:
            return []
            
        # Get source document from context
        source_document = decision_context.pending_constraints.get("clinical_note", "")
        if not source_document:
            # Try other common document keys
            source_document = decision_context.pending_constraints.get("source_document", "")
            if not source_document:
                source_document = str(decision_context.external_context)
        
        # Build ancestry path
        ancestry = [str(ancestor.id) + ": " + ancestor.name for ancestor in decision_context.ancestors]
        
        # Format candidates with descriptions
        candidate_descriptions = []
        for node_id, action in candidates:
            # We only handle GOTO actions for code selection
            if action != Action.GOTO:
                continue
                
            # Get target node info (this might need tree lookup depending on implementation)
            candidate_descriptions.append(f"{node_id}: {action.value} action")
        
        if not candidate_descriptions:
            return []
        
        try:
            # Get node-specific signature with any customizations
            active_coder = self._get_node_specific_coder(current_node.id)
            
            # Dynamic signature already handles multi-candidate optimization
            
            # Call DSPy with configured signature, using context if LM is available
            if self.lm:
                with dspy.context(lm=self.lm):
                    result = active_coder(
                        source_document=source_document,
                        current_position=f"{current_node.id}: {current_node.name}",
                        position_ancestry=ancestry,
                        candidate_codes=candidate_descriptions
                    )
            else:
                # Use default configured LM
                result = active_coder(
                    source_document=source_document,
                    current_position=f"{current_node.id}: {current_node.name}",
                    position_ancestry=ancestry,
                    candidate_codes=candidate_descriptions
                )
            
            # Extract selected codes and reasoning
            selected_codes = result.selected_codes if hasattr(result, 'selected_codes') else []
            citations = result.citations if hasattr(result, 'citations') else []
            reasoning_quality = result.reasoning_quality if hasattr(result, 'reasoning_quality') else "MODERATE"
            
            # Get reasoning from ChainOfThought
            reasoning = ""
            if hasattr(result, 'reasoning'):
                reasoning = result.reasoning
            
            # Build rich candidates
            rich_candidates = []
            for i, selected_code in enumerate(selected_codes):
                # Parse code from selection (format: "CODE: description")
                code_parts = selected_code.split(":", 1)
                if len(code_parts) >= 1:
                    target_code = code_parts[0].strip()
                    
                    # Find matching candidate
                    matching_candidate = None
                    for node_id, action in candidates:
                        if str(node_id) == target_code or target_code in str(node_id):
                            matching_candidate = (node_id, action)
                            break
                    
                    if matching_candidate:
                        node_id, action = matching_candidate
                        metadata = {
                            "selection_method": f"dspy_{self.domain_name}",
                            "rank": i + 1,
                            "dspy_reasoning": reasoning,
                            "citations": citations,
                            "reasoning_quality": reasoning_quality,
                            "signature": self.signature_class.__name__,
                            "domain": self.domain_name
                        }
                        
                        rich_candidates.append(RichCandidate(
                            target=node_id,
                            action=action,
                            metadata=metadata
                        ))
            
            # If no matches, return first candidate as fallback
            if not rich_candidates and candidates:
                node_id, action = candidates[0]
                metadata = {
                    "selection_method": f"dspy_{self.domain_name}_fallback",
                    "rank": 1,
                    "dspy_reasoning": f"Fallback selection: {reasoning}",
                    "citations": citations,
                    "reasoning_quality": "POOR",  # Fallback is always poor quality
                    "signature": self.signature_class.__name__,
                    "domain": self.domain_name
                }
                rich_candidates.append(RichCandidate(
                    target=node_id,
                    action=action,
                    metadata=metadata
                ))
            
            # Limit to max_candidates for beam search control  
            return rich_candidates[:self.max_candidates]
            
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            # Check for API/Model errors that should halt execution
            if any(keyword in error_str.lower() for keyword in [
                "ratelimiterror", "rate limit", "quota", "billing", "api",
                "authentication", "permission", "access", "key"
            ]) or "litellm" in error_str.lower():
                print(f"\n🚫 DSPy Agent Error: API/Service error detected")
                print(f"   Error Type: {error_type}")
                print(f"   Error: {error_str}")
                print(f"   💡 This usually indicates:")
                print(f"      • API rate limit exceeded")
                print(f"      • Authentication/API key issues")
                print(f"      • Service quota exhausted")
                print(f"      • Network connectivity problems")
                print(f"   🔧 Try: Wait and retry, check API keys, verify service status")
                print(f"   🛑 Halting execution - API service unavailable")
                return []  # Halt execution completely
            
            # Check for JSON adapter failures or structured output issues  
            elif any(keyword in error_str.lower() for keyword in [
                "json", "structured output format", "failed to use structured output",
                "truncated", "max_tokens", "parsing", "format"
            ]):
                print(f"\n🚫 DSPy Agent Error: Model output incompatible with agent")
                print(f"   Error Type: {error_type}")
                print(f"   Error: {error_str}")
                print(f"   💡 This usually indicates:")
                print(f"      • Model output was truncated (increase max_tokens)")
                print(f"      • Model doesn't support structured JSON output") 
                print(f"      • Model response format is incompatible")
                print(f"   🔧 Try: Different model or increase max_tokens parameter")
                print(f"   🛑 Halting execution - model incompatible with agent requirements")
                return []  # Halt execution completely
            
            # For any other errors, halt execution to prevent poor decisions
            else:
                print(f"\n🚫 DSPy Agent Error: Unexpected reasoning failure")
                print(f"   Error Type: {error_type}")
                print(f"   Error: {error_str}")
                print(f"   💡 This indicates an unexpected system error")
                print(f"   🔧 Try: Check logs, verify system state, retry with different parameters")
                print(f"   🛑 Halting execution to prevent unreliable decisions")
                return []  # Halt execution on any reasoning error
    
    def archive_exemplar(self, example: dict[str, Any]) -> None:
        """Archive training example for DSPy self-optimization."""
        example_with_domain = example.copy()
        example_with_domain["domain"] = self.domain_name
        example_with_domain["signature"] = self.signature_class.__name__
        self.training_examples.append(example_with_domain)
        
    def optimize(self, node_code: str | None = None) -> None:
        """Trigger DSPy self-optimization using collected examples."""
        if len(self.training_examples) > 0:
            print(f"🔧 DSPy optimization triggered for {self.domain_name} with {len(self.training_examples)} examples")
            # In production, implement actual DSPy optimization here
            # For now, just log that optimization would happen
        
    def get_training_stats(self) -> dict[str, Any]:
        """Get statistics about collected training data."""
        return {
            "domain": self.domain_name,
            "signature": self.signature_class.__name__,
            "examples_collected": len(self.training_examples),
            "successful_examples": len([ex for ex in self.training_examples if ex.get("successful", False)])
        }
    
    def add_rule(self, node_id: str, feedback: str | None = None, training_examples: list[dict[str, Any]] | None = None, model: str | None = None) -> bool:
        """
        Add optimization rule for DSPy agent (signature-level optimization).
        
        This method aligns with LLM agent patterns by supporting feedback that
        modifies signature field descriptions, plus training examples for future compilation.
        
        Args:
            node_id: Target node ID for optimization (e.g., "E11.22", "ROOT")
            feedback: Feedback to modify reasoning behavior (consistent with LLM agents)
                     Examples: "Focus on differential diagnosis between E11.21 vs E11.22"
                              "Emphasize systematic clinical reasoning"
            training_examples: Training examples for future DSPy compilation
            model: Optional model override for this node
            
        Returns:
            True if rule was added successfully
            
        Examples:
            # Feedback (consistent API with LLM agents)
            agent.add_rule("E11.22", "Focus on distinguishing diabetic nephropathy from chronic kidney disease")
            
            # Training examples (for future DSPy compilation)  
            agent.add_rule("ROOT", training_examples=[{"source": "...", "target": "chapter_4"}])
            
            # Combined approach
            agent.add_rule("E11.22", 
                          feedback="Use systematic differential diagnosis", 
                          training_examples=[...])
        """
        # Handle feedback by metaprompt optimization of signature field descriptions
        if feedback:
            if node_id not in self.node_rules:
                self.node_rules[node_id] = {}
                
            # Use metaprompt optimization (like LLM agents) to generate better field descriptions
            optimized_desc = self._generate_optimized_field_description(
                node_id=node_id,
                field_name="selected_codes", 
                feedback=feedback,
                model=model
            )
            self.node_rules[node_id]["selected_codes"] = optimized_desc
            
            # Also optimize reasoning quality field if available
            if hasattr(self.signature_class, "reasoning_quality"):
                reasoning_desc = self._generate_optimized_field_description(
                    node_id=node_id,
                    field_name="reasoning_quality",
                    feedback=feedback, 
                    model=model
                )
                self.node_rules[node_id]["reasoning_quality"] = reasoning_desc
            
            # Clear cached coder to force regeneration
            if node_id in self.node_coders:
                del self.node_coders[node_id]
        
        # Handle training examples for future DSPy compilation
        if training_examples:
            for example in training_examples:
                # Add context about which node this example relates to
                example_with_context = example.copy()
                example_with_context["target_node"] = node_id
                example_with_context["optimization_model"] = model
                self.training_examples.append(example_with_context)
        
        if model:
            # Store model override for future optimizations
            if not hasattr(self, '_optimization_models'):
                self._optimization_models = {}
            self._optimization_models[node_id] = model
            
        print(f"✅ Added DSPy optimization rule for node {node_id} with {len(training_examples or [])} examples")
        
        # Trigger optimization if we have enough examples
        if len(self.training_examples) >= 5:
            self.optimize(node_id)
            
        return True




# =================================================================================
# Generic Agent Factory Function
# =================================================================================

def create_configurable_agent(
    domain: str,
    provider: str = "openai", 
    model: str = "gpt-4o",
    custom_prompts: dict[str, str] | None = None,
    max_candidates: int | None = None
) -> ConfigurableDSPyAgent:
    """
    Create DSPy agent for any coding domain.
    
    Args:
        domain: Coding domain ('icd10', 'cpt', 'snomed')
        provider: LLM provider
        model: Model name
        custom_prompts: Optional domain-specific prompts
        max_candidates: Maximum candidates to select (int).
                       None defaults to 1 candidate.
        
    Returns:
        Configured DSPy agent for the specified domain
        
    Examples:
        # Single candidate (default)
        agent = create_configurable_agent("icd10", "openai", "gpt-4o", max_candidates=1)
        
        # Multiple candidates (beam search)
        agent = create_configurable_agent("icd10", "openai", "gpt-4o", max_candidates=3)
        
        # Many candidates (exhaustive search)
        agent = create_configurable_agent("icd10", "openai", "gpt-4o", max_candidates=10)
    """
    signature_map = {
        "icd10": ICD10CodingSignature,
        "cpt": CPTCodingSignature, 
        "snomed": SNOMEDCodingSignature
    }
    
    if domain not in signature_map:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(signature_map.keys())}")
    
    try:
        from core.llm import get_model_config
        config = get_model_config(provider, model)
        return ConfigurableDSPyAgent(
            signature_class=signature_map[domain],
            domain_name=domain,
            model_config=config,
            custom_prompts=custom_prompts,
            max_candidates=max_candidates
        )
    except Exception as e:
        print(f"Warning: Could not configure {provider}/{model}: {e}")
        return ConfigurableDSPyAgent(signature_map[domain], domain, custom_prompts=custom_prompts, max_candidates=max_candidates)


# Note: Domain-specific factory functions have been moved to core/domains/medical/
# Use: from core.domains.medical import icd10_domain, cpt_domain, snomed_domain
# Examples:
#   icd10_domain.create_icd10_dspy_agent("openai", "gpt-4o")
#   cpt_domain.create_cpt_dspy_agent("anthropic", "claude-sonnet-4")
#   snomed_domain.create_snomed_dspy_agent("cerebras", "llama3.1-8b")






# Example usage and testing
if __name__ == "__main__":
    from core.llm import list_providers, list_models
    
    print("DSPy Agent Configuration Demo")
    print("=" * 50)
    
    # List available providers and models
    for provider in list_providers():
        models = list_models(provider)
        print(f"\n{provider.upper()}:")
        print(f"  Models: {', '.join(models)}")
    
    print(f"\n{'=' * 50}")
    print("Usage Examples:")
    print("=" * 50)
    print("# Domain-specific agents now in separate modules")
    print('from core.domains.medical import icd10_domain, cpt_domain, snomed_domain')
    print('icd10_agent = icd10_domain.create_icd10_dspy_agent("openai", "gpt-4o")')
    print('cpt_agent = cpt_domain.create_cpt_dspy_agent("anthropic", "claude-sonnet-4")')
    print('snomed_agent = snomed_domain.create_snomed_dspy_agent("cerebras", "llama3.1-8b")')
    print("\n# Quick configuration from core engines") 
    print('from core.engines import quick_configure')
    print('lm = quick_configure("cerebras", "qwen-3-235b-a22b-thinking-2507")')
    print("\n# Manual configuration from core engines")
    print('from core.engines import create_dspy_lm')
    print('lm = create_dspy_lm("anthropic", "claude-sonnet-4-20250514")')
    print("dspy.configure(lm=lm)")
    print("\n# Configurable agent with specific domains")
    print('configurable = create_configurable_agent("icd10", "openai", "gpt-4o")')
    print("\n# Custom models with environment variables")
    print('# Set environment: CUSTOM_API_KEY=xxx, CUSTOM_API_BASE=https://api.example.com/v1')
    print('# agent = icd10_domain.create_icd10_dspy_agent("custom", "custom-llm-1")')