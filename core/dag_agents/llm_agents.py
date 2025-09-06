"""
LLM Agents for Multi-Domain Coding Systems
=============================================

This module provides agents that use direct LLM calls for different coding domains.
Configuration concerns are handled by core modules, keeping agent logic separate.
"""

from typing import Any
from core import NodeId, Action, DecisionContext, RichCandidate, CandidateAgent
# Import centralized LLM configuration utilities
from core.engines import (
    call_llm, simple_chat, create_llm_config,
    create_openai_config, create_anthropic_config, create_cerebras_config,
    validate_llm_setup
)
from typing import Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

@dataclass
class OptimizationRule:
    """Rule for optimizing prompts at specific nodes."""
    node_id: str
    feedback: str
    training_examples: list[dict[str, Any]] | None = None
    model_override: str | None = None
    created_at: str = ""
    optimized_prompt: str | None = None
    performance_data: dict[str, Any] | None = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.performance_data is None:
            self.performance_data = {"uses": 0, "success_rate": 0.0}


class PromptOptimizer:
    """Manages node-specific prompt optimization rules and persistence."""
    
    def __init__(self, storage_path: str = ".codr_llm_optimization"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.rules_file = self.storage_path / "optimization_rules.json"
        self.rules: dict[str, OptimizationRule] = {}
        self.load_rules()
    
    def add_rule(self, node_id: str, feedback: str, training_examples: list[dict[str, Any]] | None = None, model: str | None = None) -> bool:
        """
        Add optimization rule for a specific node.
        
        Args:
            node_id: Target node ID (e.g., "E11.22", "ROOT")
            feedback: Feedback for metaprompt optimization
            training_examples: Training examples that can inform the optimization (optional)
                              Format: [{"input": "clinical note", "expected": "target_code"}, ...]
            model: Model to use for this rule (e.g., "anthropic/claude-3-sonnet")
                   Defaults to agent's current model if not specified
            
        Returns:
            True if rule was added/updated successfully
        """
        rule = OptimizationRule(
            node_id=node_id,
            feedback=feedback,
            training_examples=training_examples,
            model_override=model
        )
        
        self.rules[node_id] = rule
        self.save_rules()
        return True
    
    def get_rule(self, node_id: str) -> OptimizationRule | None:
        """Get optimization rule for a specific node."""
        return self.rules.get(node_id)
    
    def has_rule(self, node_id: str) -> bool:
        """Check if optimization rule exists for node."""
        return node_id in self.rules
    
    def save_rules(self):
        """Persist optimization rules to disk."""
        rules_data = {}
        for node_id, rule in self.rules.items():
            rules_data[node_id] = {
                "node_id": rule.node_id,
                "feedback": rule.feedback,
                "model_override": rule.model_override,
                "created_at": rule.created_at,
                "optimized_prompt": rule.optimized_prompt,
                "performance_data": rule.performance_data
            }
        
        with open(self.rules_file, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def load_rules(self):
        """Load optimization rules from disk."""
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r') as f:
                    rules_data = json.load(f)
                    for node_id, rule_dict in rules_data.items():
                        self.rules[node_id] = OptimizationRule(**rule_dict)
            except Exception as e:
                print(f"Warning: Could not load optimization rules: {e}")


class LLMAgent(CandidateAgent):
    """
    Agent that uses direct LLM calls via LiteLLM.
    
    This demonstrates how the centralized provider configuration
    can be used by different agent frameworks.
    """
    
    def __init__(
        self, 
        provider: str = "openai", 
        model: str = "gpt-4o-mini", 
        domain: str = "icd10",
        max_candidates: int | None = None
    ):
        self.provider = provider
        self.model = model
        self.domain = domain
        
        # Configure candidate selection count
        if max_candidates is None:
            self.max_candidates = 1  # Default to single candidate
        else:
            self.max_candidates = max(1, int(max_candidates))  # Ensure at least 1
        
        # Initialize prompt optimizer
        self.optimizer = PromptOptimizer()
        
        # Validate LLM setup using centralized validation
        self.setup_status = validate_llm_setup(provider, model)
        if not self.setup_status.get("litellm_available", False):
            print("Warning: LiteLLM not available. Using mock responses.")
    
    def add_rule(self, node_id: str, feedback: str, training_examples: list[dict[str, Any]] | None = None, model: str | None = None) -> bool:
        """
        Add optimization rule for a specific node.
        
        This is the core agent primitive for node-level prompt optimization.
        
        Args:
            node_id: Target node ID (e.g., "E11.22", "ROOT") 
            feedback: Feedback/instruction for metaprompt optimization
            training_examples: Training examples that can inform the optimization (optional)
                              Format: [{"input": "clinical note", "expected": "target_code"}, ...]
            model: Optional model override (uses agent's model by default)
            
        Returns:
            True if rule was added successfully
            
        Example:
            agent.add_rule(
                node_id="E11.22",
                feedback="Focus on distinguishing between diabetic nephropathy (E11.21) "
                             "and chronic kidney disease (E11.22). The key is whether staging "
                             "codes like N18.x are needed for complete coding.",
                training_examples=[{"input": "T2DM with CKD stage 3", "expected": "E11.22"}],
                model="gpt-4o"  # Optional override
            )
        """
        effective_model = model if model else f"{self.provider}/{self.model}"
        return self.optimizer.add_rule(node_id, feedback, training_examples, effective_model)
    
    def select(self, decision: DecisionContext, candidates: list[tuple[NodeId, Action]]) -> list[RichCandidate]:
        """Select candidates using direct LLM calls."""
        
        if not candidates:
            return []
        
        # Get current node ID for optimization
        current_node_id = str(decision.node.id) if decision.node else "ROOT"
        
        # Build context for LLM (with potential optimization)
        context = self._build_optimized_context(decision, candidates, current_node_id)
        
        # Make LLM call using centralized configuration
        if self.setup_status.get("litellm_available", False):
            try:
                reasoning = self._call_llm(context, current_node_id)
            except Exception as e:
                print(f"LLM call failed: {e}, falling back to simple heuristics")
                reasoning = "Fallback: Using first candidate due to LLM failure"
        else:
            reasoning = "Mock: Using simple heuristics (LiteLLM not available)"
        
        # Parse LLM response to extract selections (multiple candidates)
        selected_options = self._parse_llm_selections(reasoning)
        
        # Convert to rich candidates with LLM-based scoring
        rich_candidates = []
        for i, (node_id, action) in enumerate(candidates):
            
            # Score based on LLM selections (preference order)
            if selected_options and (i + 1) in selected_options:
                # LLM selected this option - score based on preference order
                preference_rank = selected_options.index(i + 1)  # 0 = most preferred
                score = 0.95 - (preference_rank * 0.05)  # High score with slight preference penalty
                selection_method = "llm_guided"
            elif selected_options:
                # LLM didn't select this option but made selections
                score = 0.3  # Lower score for non-selected
                selection_method = "llm_guided"
            else:
                # Could not parse LLM selections, use position-based fallback
                score = max(0.8 - (i * 0.1), 0.1)
                selection_method = "heuristic_fallback"
            
            rich_candidates.append(RichCandidate(
                target=node_id,
                action=action,
                metadata={
                    "agent_type": "raw_llm",
                    "provider": self.provider,
                    "model": self.model,
                    "domain": self.domain,
                    "score": score,
                    "reasoning": reasoning,
                    "selection_method": selection_method,
                    "llm_selected_options": selected_options,
                    "candidate_position": i + 1,
                    "preference_rank": selected_options.index(i + 1) if selected_options and (i + 1) in selected_options else None
                }
            ))
        
        # Sort by score (descending) to ensure LLM selections come first in preference order
        rich_candidates.sort(key=lambda x: x.metadata["score"], reverse=True)
        
        # Limit to max_candidates for beam search control
        return rich_candidates[:self.max_candidates]
    
    def _parse_llm_selections(self, reasoning: str) -> list[int]:
        """Parse LLM response to extract selected option numbers in order of preference."""
        if not reasoning:
            return []
        
        import re
        
        selections = []
        
        # Try to find "SELECTIONS: N, M, P" format first (new multi-select format)
        selections_match = re.search(r'SELECTIONS?\s*:\s*([0-9,\s]+)', reasoning, re.IGNORECASE)
        if selections_match:
            try:
                selections_str = selections_match.group(1)
                selections = [int(x.strip()) for x in selections_str.split(',') if x.strip().isdigit()]
                if selections:
                    return selections[:self.max_candidates]  # Limit to requested count
            except ValueError:
                pass
        
        # Fallback: Try to find "SELECTION: N" format (legacy single select)
        selection_match = re.search(r'SELECTION:\s*(\d+)', reasoning, re.IGNORECASE)
        if selection_match:
            try:
                return [int(selection_match.group(1))]
            except ValueError:
                pass
        
        # Try to find other patterns like "option 4", "choice 2", "select 3", etc.
        patterns = [
            r'option\s+(\d+)',
            r'choice\s+(\d+)', 
            r'select\s+(\d+)',
            r'choose\s+(\d+)',
            r'pick\s+(\d+)',
            r'prefer\s+(\d+)',
            r'#(\d+)',
            r'(\d+)\.',  # Match "4." at start of line
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match)
                    if num not in all_matches:  # Avoid duplicates
                        all_matches.append(num)
                except ValueError:
                    continue
        
        # Return up to max_candidates selections
        return all_matches[:self.max_candidates] if all_matches else []
    
    def _build_optimized_context(self, decision: DecisionContext, candidates: list[tuple], node_id: str) -> str:
        """Build context with potential node-specific optimization."""
        # Check if we have an optimization rule for this node
        rule = self.optimizer.get_rule(node_id)
        
        if rule and rule.optimized_prompt:
            # Use cached optimized prompt
            return self._build_context_with_optimized_prompt(decision, candidates, rule.optimized_prompt)
        elif rule:
            # Generate optimized prompt using metaprompt
            optimized_prompt = self._generate_optimized_prompt(decision, candidates, node_id, rule)
            rule.optimized_prompt = optimized_prompt
            self.optimizer.save_rules()
            return self._build_context_with_optimized_prompt(decision, candidates, optimized_prompt)
        else:
            # Use standard context building
            return self._build_context(decision, candidates)
    
    def _build_context(self, decision: DecisionContext, candidates: list[tuple]) -> str:
        """Build rich context string for LLM with clinical information and detailed candidate descriptions."""
        
        # Extract clinical context from DecisionContext
        agent_inputs = decision.external_context.get("agent_inputs", {})
        clinical_note = agent_inputs.get("clinical_context", "")
        
        # Handle nested agent_inputs structure: {'agent_inputs': {'clinical_context': '...'}}
        if not clinical_note and "agent_inputs" in agent_inputs:
            nested_inputs = agent_inputs.get("agent_inputs", {})
            clinical_note = nested_inputs.get("clinical_context", "")
        
        # Fallback to pending_constraints for clinical data
        if not clinical_note:
            clinical_note = decision.pending_constraints.get("clinical_note", "")
        
        # Extract feedback and provenance information for enhanced context
        feedback_info = ""
        provenance_info = ""
        
        # Check for feedback from rewind operations
        if hasattr(decision.node, 'ctx') and decision.node.ctx and hasattr(decision.node.ctx, 'data'):
            ctx_data = decision.node.ctx.data
        else:
            # Try to get context from other sources
            ctx_data = decision.pending_constraints
            
        if "feedback" in ctx_data or "rewind_feedback" in ctx_data:
            feedback = ctx_data.get("feedback") or ctx_data.get("rewind_feedback", {})
            if feedback:
                reason = feedback.get("reason", "Previous selection correction")
                correction = feedback.get("correction", "")
                feedback_info = f"\n⚠️  FEEDBACK FROM PREVIOUS ATTEMPT:\n{reason}\n"
                if correction:
                    feedback_info += f"Correction needed: {correction}\n"
        
        # Check for code provenance information
        if "previous_path" in ctx_data:
            previous_path = ctx_data.get("previous_path", [])
            rewind_from = ctx_data.get("rewind_from", "")
            if previous_path:
                provenance_info = f"\n📋 CODE PROVENANCE:\n"
                provenance_info += f"Previous path taken: {' → '.join(previous_path)}\n"
                if rewind_from:
                    provenance_info += f"Rewound from: {rewind_from}\n"
                provenance_info += "Use this information to avoid repeating incorrect decisions.\n"
        
        clinical_context = ""
        if clinical_note:
            clinical_context = f"\n🏥 CLINICAL SCENARIO:\n{clinical_note.strip()}\n"
        
        # Build current position and ancestry information
        current_info = f"📍 Current Position: {decision.node.name}" if decision.node else "📍 Current Position: ROOT"
        
        ancestry_info = ""
        if decision.path and len(decision.path) > 1:
            path_names = []
            for node_id in decision.path:
                # Try to get tree to resolve node names
                try:
                    from core.domains.medical.trees.icd_tree import create_navigator
                    tree = create_navigator()
                    node = tree.get(node_id)
                    path_names.append(node.name if node else str(node_id))
                except:
                    path_names.append(str(node_id))
            ancestry_info = f"\n🗂️  Path taken: {' → '.join(path_names)}\n"
        
        # Build rich candidate information
        candidates_info = "\n🎯 AVAILABLE OPTIONS:\n"
        
        # Check if these are chapter candidates (ROOT level selection)
        if str(decision.node.id) == "ROOT" if decision.node else False:
            # Chapter selection - provide rich descriptions
            try:
                from core.domains.medical.trees.icd_tree import create_navigator, get_chapters_for_selection
                navigator = create_navigator()
                chapters = get_chapters_for_selection(navigator)
                
                for i, (node_id, action) in enumerate(candidates):
                    if str(node_id).startswith("chapter_"):
                        chapter_num = int(str(node_id).replace("chapter_", ""))
                        if 1 <= chapter_num <= len(chapters):
                            chapter = chapters[chapter_num - 1]
                            candidates_info += f"{i+1}. {action.value.upper()} → {node_id}: {chapter['name']}\n"
                        else:
                            candidates_info += f"{i+1}. {action.value.upper()} → {node_id}\n"
                    else:
                        candidates_info += f"{i+1}. {action.value.upper()} → {node_id}\n"
            except Exception:
                # Fallback to basic candidate display
                for i, (node_id, action) in enumerate(candidates[:10]):
                    candidates_info += f"{i+1}. {action.value.upper()} → {node_id}\n"
        else:
            # Regular node selection - show detailed code information
            for i, (node_id, action) in enumerate(candidates[:10]):
                try:
                    from core.domains.medical.trees.icd_tree import create_navigator
                    tree = create_navigator()
                    node = tree.get(node_id)
                    if node:
                        candidates_info += f"{i+1}. {action.value.upper()} → {node_id}: {node.name}\n"
                    else:
                        candidates_info += f"{i+1}. {action.value.upper()} → {node_id}\n"
                except:
                    candidates_info += f"{i+1}. {action.value.upper()} → {node_id}\n"
        
        if len(candidates) > 10:
            candidates_info += f"... and {len(candidates) - 10} more options available\n"
        
        # Build comprehensive context
        context = f"""
{clinical_context}
{feedback_info}
{provenance_info}
{current_info}
{ancestry_info}
{candidates_info}

🎯 TASK: Analyze the clinical scenario and select the most appropriate option(s) for {self.domain} coding.

📋 INSTRUCTIONS:
- Consider the clinical presentation, symptoms, conditions mentioned
- Match the scenario to the most relevant medical category/code
- If feedback is provided, learn from previous mistakes and avoid repeating them
- If code provenance is available, use it to understand the decision path
- Provide your clinical reasoning
- Select your top {self.max_candidates} option(s) in order of preference
- Explain why these choices best fit the clinical evidence

🤖 Please respond with:
REASONING: [Your clinical analysis]
SELECTIONS: [Comma-separated option numbers in preference order, e.g., "3, 1, 5"]
"""
        
        return context
    
    def _call_llm(self, context: str, _node_id: str = "") -> str:
        """Make LLM call using centralized configuration."""
        if not self.setup_status.get("litellm_available", False):
            return "Mock response: Selected first candidate"
        
        # Use centralized LLM calling utility
        messages = [
            {"role": "system", "content": f"You are a {self.domain} coding assistant."},
            {"role": "user", "content": context}
        ]
        
        response = call_llm(
            messages=messages,
            provider=self.provider,
            model=self.model,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _generate_optimized_prompt(self, decision: DecisionContext, candidates: list[tuple], node_id: str, rule: OptimizationRule) -> str:
        """Generate optimized prompt using metaprompt and rule feedback."""
        
        # Build comprehensive metaprompt context
        node_name = decision.node.name if decision.node else "Root"
        current_prompt = self._build_context(decision, candidates)  # Get current standard prompt
        
        # Format candidates for metaprompt
        candidates_info = []
        for i, (candidate_id, action) in enumerate(candidates):
            candidates_info.append(f"{i+1}. {action.upper()} → {candidate_id}")
        
        # Check for additional injected nodes from rules (if system supports it)
        injected_nodes_info = self._get_injected_nodes_info(decision, node_id)
        
        metaprompt = f"""
PROMPT OPTIMIZATION REQUEST

INSTRUCTION: Optimize the medical coding decision prompt for this specific node to improve agent performance.

CURRENT NODE: {node_id} - {node_name}

AVAILABLE CANDIDATES:
{chr(10).join(candidates_info)}

{injected_nodes_info}

CURRENT DEPLOYED PROMPT:
{current_prompt}

RULE FEEDBACK:
{rule.feedback}

OPTIMIZATION REQUIREMENTS:
1. Create an improved prompt that helps the agent make better decisions at this specific node
2. Address the specific issues mentioned in the rule feedback  
3. Maintain the same input/output format (clinical context → REASONING/SELECTION)
4. Focus on key differentiators between the available candidates
5. Include domain-specific medical knowledge relevant to this node

Please generate an optimized prompt that will improve decision accuracy at node {node_id}.
"""
        
        # Use the specified model for optimization or fall back to agent's model
        optimization_model = rule.model_override or f"{self.provider}/{self.model}"
        
        try:
            # Make optimization call
            messages = [
                {"role": "system", "content": "You are a prompt optimization expert specializing in medical coding decision support."},
                {"role": "user", "content": metaprompt}
            ]
            
            # Parse provider/model if it contains "/"
            if "/" in optimization_model:
                opt_provider, opt_model = optimization_model.split("/", 1)
            else:
                opt_provider, opt_model = self.provider, optimization_model
            
            response = call_llm(
                messages=messages,
                provider=opt_provider,
                model=opt_model,
                max_tokens=1500,
                temperature=0.3  # Lower temperature for consistent optimization
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Warning: Optimization failed for {node_id}: {e}")
            return current_prompt  # Fallback to current prompt
    
    def _build_context_with_optimized_prompt(self, decision: DecisionContext, candidates: list[tuple], optimized_prompt: str) -> str:
        """Build context using an optimized prompt template."""
        # Extract dynamic content
        clinical_context = self._extract_clinical_context(decision)
        current_info = f"📍 Current Position: {decision.node.name}" if decision.node else "📍 Current Position: ROOT"
        candidates_info = self._format_candidates_for_context(candidates)
        
        # Replace placeholders in optimized prompt
        context = optimized_prompt
        context = context.replace("{clinical_context}", clinical_context)
        context = context.replace("{current_info}", current_info)  
        context = context.replace("{candidates_info}", candidates_info)
        
        return context
    
    def _extract_clinical_context(self, decision: DecisionContext) -> str:
        """Extract clinical context from DecisionContext."""
        agent_inputs = decision.external_context.get("agent_inputs", {})
        clinical_note = agent_inputs.get("clinical_context", "")
        
        if not clinical_note:
            clinical_note = decision.pending_constraints.get("clinical_note", "")
        
        if clinical_note:
            return f"\n🏥 CLINICAL SCENARIO:\n{clinical_note.strip()}\n"
        return ""
    
    def _format_candidates_for_context(self, candidates: list[tuple]) -> str:
        """Format candidates for prompt context."""
        candidates_info = "\n🎯 AVAILABLE OPTIONS:\n"
        
        for i, (node_id, action) in enumerate(candidates[:10]):
            try:
                from core.domains.medical.trees.icd_tree import create_navigator
                tree = create_navigator()
                node = tree.get(node_id)
                if node:
                    candidates_info += f"{i+1}. {action.value.upper()} → {node_id}: {node.name}\n"
                else:
                    candidates_info += f"{i+1}. {action.value.upper()} → {node_id}\n"
            except:
                candidates_info += f"{i+1}. {action.value.upper()} → {node_id}\n"
        
        if len(candidates) > 10:
            candidates_info += f"... and {len(candidates) - 10} more options available\n"
        
        return candidates_info
    
    def _get_injected_nodes_info(self, decision: DecisionContext, node_id: str) -> str:
        """
        Get information about any nodes injected by rules (placeholder for future functionality).
        
        Args:
            decision: Current decision context (used for future rule-based node injection)
            node_id: Current node ID (used for future rule-based node injection)
        """
        # This is a placeholder for future functionality where rules might inject additional candidates
        # Future implementation will analyze decision.node and node_id to determine injected candidates
        _ = decision, node_id  # Explicitly mark as intentionally unused for now
        return ""


def create_llm_agent(
    provider: str = "openai", 
    model: str = "gpt-4o-mini", 
    domain: str = "icd10",
    max_candidates: int | None = None
) -> LLMAgent:
    """
    Create an LLM agent using centralized configuration.
    
    Args:
        provider: LLM provider (openai, anthropic, etc.)
        model: Model name
        domain: Domain for coding (icd10, cpt, snomed)
        max_candidates: Maximum candidates to select (int).
                       None defaults to 1 candidate.
                       
    Examples:
        # Single candidate (default)
        agent = create_llm_agent("openai", "gpt-4o", max_candidates=1)
        
        # Multiple candidates (beam search)
        agent = create_llm_agent("openai", "gpt-4o", max_candidates=3)
        
        # Many candidates (exhaustive search)
        agent = create_llm_agent("openai", "gpt-4o", max_candidates=10)
    """
    return LLMAgent(provider=provider, model=model, domain=domain, max_candidates=max_candidates)


# Note: Domain-specific factory functions have been moved to core/domains/medical/
# Use: from core.domains.medical import icd10_domain, cpt_domain, snomed_domain
# Examples:
#   icd10_domain.create_icd10_llm_agent("openai", "gpt-4o")
#   cpt_domain.create_cpt_llm_agent("anthropic", "claude-sonnet-4")
#   snomed_domain.create_snomed_llm_agent("cerebras", "llama3.1-8b")


# Example usage and comparison
if __name__ == "__main__":
    print("LLM Agent Demo")
    print("=" * 50)
    
    try:
        # Create generic LLM agent
        generic_agent = create_llm_agent("openai", "gpt-4o-mini", "icd10")
        print(f"✅ Created generic LLM agent: {generic_agent.provider}/{generic_agent.model}")
        
        # Domain-specific agents are now in separate modules
        print(f"\n📁 Domain-specific agents moved to:")
        print(f"   from core.domains.medical import icd10_domain")
        print(f"   icd10_agent = icd10_domain.create_icd10_llm_agent('openai', 'gpt-4o')")
        print(f"   cpt_agent = cpt_domain.create_cpt_llm_agent('anthropic', 'claude-sonnet-4')")
        print(f"   snomed_agent = snomed_domain.create_snomed_llm_agent('cerebras', 'llama3.1-8b')")
        
        print(f"\n🔧 Clean Architecture Benefits:")
        print(f"• Core LLM agent logic separated from domain configuration")
        print(f"• Domain-specific factories organized by coding system")
        print(f"• Easy extension: add new domains without modifying core files")
        print(f"• Same provider configs used across all frameworks")
        
        # Check setup status
        setup_status = validate_llm_setup("openai", "gpt-4o-mini")
        if not setup_status.get("litellm_available", False):
            print(f"\n⚠️  LiteLLM not installed - agents will use fallback logic")
            print(f"   Install with: pip install litellm")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()