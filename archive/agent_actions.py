"""
Agent Actions for Clinical Decision Engine
==========================================

This module handles LLM-based actions, post-processing logic,
and complex action orchestration for clinical coding workflows.
"""

import time
from typing import Any
from agent_states import ClinicalTraversalStream, TrajectoryDecision


class ClinicalActionOrchestrator:
    """
    Orchestrates LLM-based actions and handles complex multi-step workflows.
    Manages action proposals, validation, execution planning, and post-processing.
    """
    
    def __init__(self, stream: ClinicalTraversalStream):
        self.stream = stream
    
    def propose_action(self, target_code: str, action_type: str, llm_reasoning: dict, clinical_citations: list[str] | None = None) -> dict[str, Any]:
        """
        Propose a single deterministic action based on LLM reasoning.
        This reports what will happen before committing to the action.
        """
        # Validate the proposed action
        if action_type != 'stop_question' and not self.stream._is_valid_move(target_code):
            raise ValueError(f"Proposed action invalid: {self.stream._get_block_reason(target_code)}")
        
        # Create the proposed action
        proposed_action = {
            'action_id': f"action_{len(self.stream.pending_actions)}",
            'type': action_type,  # 'navigate_child', 'execute_note_action', 'complete_leaf', 'create_parallel', 'stop_question', 'add_7th_character'
            'target_code': target_code,
            'llm_reasoning': llm_reasoning,
            'clinical_citations': clinical_citations or [],
            'deterministic_effects': self._predict_action_effects(target_code, action_type),
            'constraints_applied': self._get_constraints_to_apply(target_code),
            'dependencies': [],  # Other actions this depends on
            'priority': 1,  # Execution priority (1=highest)
            'timestamp': time.time()
        }
        
        # Add to pending actions
        self.stream.pending_actions.append(proposed_action)
        return proposed_action
    
    def propose_multiple_actions(self, action_proposals: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Propose multiple simultaneous actions with dependency management.
        
        Args:
            action_proposals: List of dicts with keys: target_code, action_type, llm_reasoning, clinical_citations, priority, dependencies
        
        Example:
            actions = [
                {
                    'target_code': 'E10.2',
                    'action_type': 'navigate_child', 
                    'llm_reasoning': {...},
                    'priority': 1
                },
                {
                    'target_code': 'E10.21',
                    'action_type': 'navigate_child',
                    'llm_reasoning': {...},
                    'dependencies': ['batch_action_0'],  # Depends on first action
                    'priority': 2
                },
                {
                    'target_code': 'C25',
                    'action_type': 'create_parallel',
                    'llm_reasoning': {...},
                    'priority': 1
                }
            ]
        """
        proposed_actions = []
        
        for i, proposal in enumerate(action_proposals):
            # Validate required fields
            if 'target_code' not in proposal or 'action_type' not in proposal:
                raise ValueError(f"Action {i}: missing required fields 'target_code' or 'action_type'")
            
            # Create action with unique ID
            action_id = f"batch_action_{len(self.stream.pending_actions) + i}"
            
            proposed_action = {
                'action_id': action_id,
                'type': proposal['action_type'],
                'target_code': proposal['target_code'],
                'llm_reasoning': proposal.get('llm_reasoning', {}),
                'clinical_citations': proposal.get('clinical_citations', []),
                'deterministic_effects': self._predict_action_effects(proposal['target_code'], proposal['action_type']),
                'constraints_applied': self._get_constraints_to_apply(proposal['target_code']),
                'dependencies': proposal.get('dependencies', []),
                'priority': proposal.get('priority', 1),
                'timestamp': time.time()
            }
            
            # Validate if not stop_question
            if proposal['action_type'] != 'stop_question' and not self.stream._is_valid_move(proposal['target_code']):
                raise ValueError(f"Action {i} ({action_id}) invalid: {self.stream._get_block_reason(proposal['target_code'])}")
            
            proposed_actions.append(proposed_action)
        
        # Add all actions to pending list
        self.stream.pending_actions.extend(proposed_actions)
        
        # Update dependencies map
        for action in proposed_actions:
            if action['dependencies']:
                self.stream.action_dependencies[action['action_id']] = action['dependencies']
        
        return {
            'proposed_actions': proposed_actions,
            'execution_plan': self._create_execution_plan(),
            'total_actions': len(proposed_actions)
        }
    
    def commit_pending_actions(self) -> dict[str, Any]:
        """
        Commit all pending actions in dependency order.
        Returns the new state after all actions are executed.
        """
        if not self.stream.pending_actions:
            raise ValueError("No pending actions to commit")
        
        # Create execution plan
        execution_plan = self._create_execution_plan()
        
        # Execute actions in planned order
        executed_actions = []
        parallel_streams_to_create = []
        
        for phase in execution_plan['phases']:
            for action in phase['actions']:
                try:
                    # Record the decision in reasoning chain
                    decision = TrajectoryDecision(
                        node_code=action['target_code'],
                        clinical_context=action['clinical_citations'],
                        llm_reasoning=action['llm_reasoning'],
                        selected_action=action
                    )
                    self.stream.reasoning_chain.append(decision)
                    
                    # Execute based on action type
                    if action['type'] in ['navigate_child', 'execute_note_action']:
                        self.stream.navigate_to(action['target_code'], validate_constraints=False)
                        
                    elif action['type'] == 'add_7th_character':
                        self._add_7th_character_extension(action['target_code'])
                        
                    elif action['type'] == 'create_parallel':
                        parallel_streams_to_create.append(action)
                        
                    elif action['type'] == 'complete_leaf':
                        # Mark stream as complete
                        pass
                        
                    elif action['type'] == 'stop_question':
                        # Set flag indicating LLM needs more information
                        break  # Stop processing further actions
                    
                    executed_actions.append(action)
                    
                except Exception as e:
                    # Action execution failed
                    action['execution_error'] = str(e)
                    executed_actions.append(action)
        
        # Clear pending actions and dependencies
        self.stream.pending_actions = []
        self.stream.action_dependencies = {}
        
        # Return results
        return {
            'new_state': self.stream.get_current_state(),
            'executed_actions': executed_actions,
            'parallel_streams_to_create': parallel_streams_to_create,
            'execution_summary': execution_plan
        }
    
    def commit_pending_action(self) -> dict[str, Any]:
        """Backwards compatibility: commit first pending action only."""
        if not self.stream.pending_actions:
            raise ValueError("No pending actions to commit")
        
        # Take only the first action
        first_action = self.stream.pending_actions[0]
        self.stream.pending_actions = [first_action]
        
        # Use the full commit method
        result = self.commit_pending_actions()
        return result['new_state']
    
    def _create_execution_plan(self) -> dict[str, Any]:
        """Create an execution plan for pending actions based on dependencies and priorities."""
        if not self.stream.pending_actions:
            return {'phases': [], 'total_actions': 0}
        
        # Sort actions by priority and dependencies
        sorted_actions = sorted(self.stream.pending_actions, key=lambda x: (x['priority'], x['timestamp']))
        
        # Create execution phases based on dependencies
        phases = []
        remaining_actions = sorted_actions.copy()
        executed_action_ids = set()
        
        while remaining_actions:
            # Find actions that can be executed (no unresolved dependencies)
            ready_actions = []
            
            for action in remaining_actions:
                dependencies = action.get('dependencies', [])
                if all(dep_id in executed_action_ids for dep_id in dependencies):
                    ready_actions.append(action)
            
            if not ready_actions:
                # Circular dependency or invalid dependency - break the cycle
                ready_actions = [remaining_actions[0]]  # Take first action to break deadlock
            
            # Create phase
            phases.append({
                'phase_id': len(phases),
                'actions': ready_actions,
                'can_execute_parallel': all(action['type'] != 'navigate_child' for action in ready_actions)
            })
            
            # Update tracking
            for action in ready_actions:
                executed_action_ids.add(action['action_id'])
                remaining_actions.remove(action)
        
        return {
            'phases': phases,
            'total_actions': len(sorted_actions),
            'estimated_parallel_streams': len([a for a in sorted_actions if a['type'] == 'create_parallel'])
        }
    
    def _add_7th_character_extension(self, seventh_char: str):
        """Add 7th character extension to current code."""
        if not self.stream.current_node:
            raise ValueError("No current node for 7th character extension")
        
        current_code = self.stream.current_node.code
        extended_code = f"{current_code}{seventh_char}"
        
        # Check if extended code exists
        extended_node = self.stream.navigator.find_by_code(extended_code)
        if not extended_node:
            # Create virtual extended code representation
            self.stream.final_suffix_codes.append(extended_code)
        else:
            # Navigate to the extended code
            self.stream.navigate_to(extended_code, validate_constraints=False)
    
    def _predict_action_effects(self, target_code: str, action_type: str) -> dict[str, Any]:
        """Predict the effects of taking an action before committing."""
        target_node = self.stream.navigator.find_by_code(target_code)
        if not target_node:
            return {'error': 'Target code not found'}
        
        effects = {
            'new_path': self.stream.path + [target_code],
            'new_constraints': {},
            'blocked_codes_added': [],
            'required_codes_added': [],
            'parallel_branches_created': [],
            'completion_status_change': False
        }
        
        # Predict constraint changes from target node notes
        target_notes = getattr(target_node, 'notes', {})
        
        if 'excludes1' in target_notes:
            blocked = [code for code, _ in target_notes['excludes1'] if code]
            effects['blocked_codes_added'].extend(blocked)
            effects['new_constraints']['excludes1'] = blocked
        
        if 'codeFirst' in target_notes:
            required = [code for code, _ in target_notes['codeFirst'] if code]
            effects['required_codes_added'].extend(required)
            effects['new_constraints']['codeFirst'] = required
        
        if 'codeAlso' in target_notes:
            branches = [code for code, _ in target_notes['codeAlso'] if code]
            effects['parallel_branches_created'].extend(branches)
            effects['new_constraints']['codeAlso'] = branches
        
        # Predict completion status
        target_is_leaf = len(target_node.children) == 0
        has_final_codes = 'sevenChrNote' in target_notes
        effects['completion_status_change'] = target_is_leaf or has_final_codes
        
        return effects
    
    def _get_constraints_to_apply(self, target_code: str) -> dict[str, Any]:
        """Get constraints that will be applied when moving to target code."""
        target_node = self.stream.navigator.find_by_code(target_code)
        if not target_node:
            return {}
        
        constraints = {}
        target_notes = getattr(target_node, 'notes', {})
        
        for note_type in ['excludes1', 'excludes2', 'codeFirst', 'useAdditionalCode', 'codeAlso', 'sevenChrNote']:
            if note_type in target_notes:
                codes = [code for code, _ in target_notes[note_type] if code]
                if codes:
                    constraints[note_type] = codes
        
        return constraints


class LLMDecisionProcessor:
    """
    Processes LLM outputs and converts them into structured clinical actions.
    Handles LLM response parsing, validation, and action generation.
    """
    
    def __init__(self, orchestrator: ClinicalActionOrchestrator):
        self.orchestrator = orchestrator
        self.stream = orchestrator.stream
    
    def process_llm_decision(self, llm_response: dict[str, Any], clinical_note: str) -> dict[str, Any]:
        """
        Process LLM decision response and convert to executable actions.
        
        Args:
            llm_response: LLM output with reasoning and proposed actions
            clinical_note: Clinical context that guided the decision
        
        Returns:
            Processed actions with validation results
        """
        # Set clinical context
        self.stream.set_clinical_context(clinical_note)
        
        # Extract LLM reasoning
        llm_reasoning = self._extract_llm_reasoning(llm_response)
        
        # Extract proposed actions
        proposed_actions = self._extract_proposed_actions(llm_response, llm_reasoning)
        
        # Validate actions against current state
        validated_actions = self._validate_proposed_actions(proposed_actions)
        
        # Generate execution plan
        if validated_actions['valid_actions']:
            try:
                result = self.orchestrator.propose_multiple_actions(validated_actions['valid_actions'])
                return {
                    'status': 'success',
                    'actions_proposed': len(validated_actions['valid_actions']),
                    'execution_plan': result['execution_plan'],
                    'invalid_actions': validated_actions['invalid_actions'],
                    'llm_reasoning': llm_reasoning
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e),
                    'llm_reasoning': llm_reasoning,
                    'proposed_actions': proposed_actions
                }
        else:
            return {
                'status': 'no_valid_actions',
                'invalid_actions': validated_actions['invalid_actions'],
                'llm_reasoning': llm_reasoning
            }
    
    def _extract_llm_reasoning(self, llm_response: dict[str, Any]) -> dict[str, Any]:
        """Extract structured reasoning from LLM response."""
        return {
            'clinical_analysis': llm_response.get('clinical_analysis', ''),
            'code_justification': llm_response.get('code_justification', ''),
            'rule_compliance': llm_response.get('rule_compliance', ''),
            'confidence': llm_response.get('confidence', 0.0),
            'alternative_options': llm_response.get('alternatives', []),
            'risk_factors': llm_response.get('risk_factors', []),
            'timestamp': time.time()
        }
    
    def _extract_proposed_actions(self, llm_response: dict[str, Any], llm_reasoning: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract proposed actions from LLM response."""
        actions = []
        
        # Handle different LLM response formats
        if 'actions' in llm_response:
            # Structured action list
            for action_data in llm_response['actions']:
                actions.append({
                    'target_code': action_data.get('code', ''),
                    'action_type': action_data.get('type', 'navigate_child'),
                    'llm_reasoning': llm_reasoning,
                    'clinical_citations': action_data.get('citations', []),
                    'priority': action_data.get('priority', 1),
                    'dependencies': action_data.get('dependencies', [])
                })
        
        elif 'target_code' in llm_response:
            # Single action response
            actions.append({
                'target_code': llm_response['target_code'],
                'action_type': llm_response.get('action_type', 'navigate_child'),
                'llm_reasoning': llm_reasoning,
                'clinical_citations': llm_response.get('clinical_citations', []),
                'priority': 1,
                'dependencies': []
            })
        
        elif 'decision' in llm_response:
            # Decision-based response
            decision = llm_response['decision']
            if decision == 'navigate':
                actions.append({
                    'target_code': llm_response.get('selected_code', ''),
                    'action_type': 'navigate_child',
                    'llm_reasoning': llm_reasoning,
                    'clinical_citations': llm_response.get('reasoning_sources', []),
                    'priority': 1,
                    'dependencies': []
                })
            elif decision == 'stop':
                actions.append({
                    'target_code': 'STOP',
                    'action_type': 'stop_question',
                    'llm_reasoning': llm_reasoning,
                    'clinical_citations': llm_response.get('questions_needed', []),
                    'priority': 1,
                    'dependencies': []
                })
        
        return actions
    
    def _validate_proposed_actions(self, proposed_actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate proposed actions against current stream state."""
        valid_actions = []
        invalid_actions = []
        
        for action in proposed_actions:
            validation_result = self._validate_single_action(action)
            
            if validation_result['valid']:
                valid_actions.append(action)
            else:
                invalid_actions.append({
                    'action': action,
                    'validation_error': validation_result['error'],
                    'suggested_fix': validation_result.get('suggested_fix', None)
                })
        
        return {
            'valid_actions': valid_actions,
            'invalid_actions': invalid_actions,
            'total_proposed': len(proposed_actions)
        }
    
    def _validate_single_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Validate a single action against current constraints."""
        target_code = action.get('target_code', '')
        action_type = action.get('action_type', '')
        
        # Basic validation
        if not target_code:
            return {'valid': False, 'error': 'Missing target_code'}
        
        if not action_type:
            return {'valid': False, 'error': 'Missing action_type'}
        
        # Special handling for stop actions
        if action_type == 'stop_question':
            return {'valid': True}
        
        # Check if target code exists
        target_node = self.stream.navigator.find_by_code(target_code)
        if not target_node:
            return {
                'valid': False, 
                'error': f'Target code {target_code} not found',
                'suggested_fix': 'Use a valid ICD-10-CM code'
            }
        
        # Check if move is valid given current constraints
        if not self.stream._is_valid_move(target_code):
            block_reason = self.stream._get_block_reason(target_code)
            return {
                'valid': False,
                'error': f'Move to {target_code} blocked: {block_reason}',
                'suggested_fix': 'Choose a different target code or resolve constraint conflict'
            }
        
        # Action type specific validation
        if action_type == 'navigate_child':
            # Check if target is actually a child of current node
            if target_node not in self.stream.current_node.children:
                # Check if it's a descendant
                is_descendant = False
                for child in self.stream.current_node.children:
                    if target_code in [desc.code for desc in child.descendants]:
                        is_descendant = True
                        break
                
                if not is_descendant:
                    return {
                        'valid': False,
                        'error': f'{target_code} is not a child or descendant of current node',
                        'suggested_fix': 'Navigate through proper hierarchy or use execute_note_action'
                    }
        
        elif action_type == 'add_7th_character':
            # Check if current code can accept 7th character extension
            current_code = self.stream.current_node.code
            if len(current_code) != 6:  # Should be 6 characters for 7th extension
                return {
                    'valid': False,
                    'error': f'Current code {current_code} has {len(current_code)} characters, needs 6 for 7th extension',
                    'suggested_fix': 'Navigate to a 6-character code first'
                }
        
        return {'valid': True}


class ClinicalPostProcessor:
    """
    Handles post-processing of clinical coding results.
    Validates completeness, generates reports, and prepares final output.
    """
    
    def __init__(self, stream: ClinicalTraversalStream):
        self.stream = stream
    
    def generate_completion_report(self) -> dict[str, Any]:
        """Generate comprehensive completion report for the clinical coding session."""
        completion_status = self.stream.get_completion_requirements()
        decision_summary = self.stream.get_decision_summary()
        
        # Analyze coding completeness
        completeness_analysis = self._analyze_completeness()
        
        # Generate final codes
        final_codes = self._generate_final_codes()
        
        # Validate against clinical guidelines
        validation_results = self._validate_clinical_guidelines()
        
        return {
            'session_summary': {
                'stream_id': self.stream.stream_id,
                'start_code': self.stream.start_code,
                'final_code': self.stream.current_node.code if self.stream.current_node else None,
                'path_taken': ' → '.join(self.stream.path),
                'total_decisions': len(self.stream.reasoning_chain),
                'session_complete': completion_status['is_complete']
            },
            'final_codes': final_codes,
            'completeness_analysis': completeness_analysis,
            'validation_results': validation_results,
            'clinical_reasoning_chain': [
                {
                    'step': i + 1,
                    'code': decision.node_code,
                    'reasoning': decision.llm_reasoning,
                    'clinical_evidence': decision.clinical_context
                }
                for i, decision in enumerate(self.stream.reasoning_chain)
            ],
            'recommendations': self._generate_recommendations(),
            'timestamp': time.time()
        }
    
    def _analyze_completeness(self) -> dict[str, Any]:
        """Analyze the completeness of the current coding session."""
        requirements = self.stream.get_completion_requirements()
        
        analysis = {
            'is_complete': requirements['is_complete'],
            'missing_requirements': [],
            'completeness_score': 0.0,
            'critical_gaps': []
        }
        
        total_requirements = 0
        met_requirements = 0
        
        # Check prefix codes
        if requirements['prefix_codes_needed']:
            total_requirements += 1
            analysis['missing_requirements'].append({
                'type': 'prefix_codes',
                'needed': requirements['prefix_codes_needed'],
                'description': 'Required codes that must be assigned before this diagnosis'
            })
        else:
            met_requirements += 1
        
        # Check suffix codes  
        if requirements['suffix_codes_needed']:
            total_requirements += 1
            analysis['missing_requirements'].append({
                'type': 'suffix_codes',
                'needed': requirements['suffix_codes_needed'],
                'description': 'Additional codes required to accompany this diagnosis'
            })
        else:
            met_requirements += 1
        
        # Check if at leaf node
        is_leaf = len(self.stream.current_node.children) == 0 if self.stream.current_node else False
        if not is_leaf:
            total_requirements += 1
            analysis['missing_requirements'].append({
                'type': 'specificity',
                'needed': [child.code for child in self.stream.current_node.children[:3]],
                'description': 'More specific coding may be available'
            })
        else:
            met_requirements += 1
        
        # Check final suffix codes
        if requirements['final_codes_available'] and not requirements['final_codes_available']:
            analysis['critical_gaps'].append('7th character extension may be required')
        
        # Calculate completeness score
        if total_requirements > 0:
            analysis['completeness_score'] = met_requirements / total_requirements
        else:
            analysis['completeness_score'] = 1.0
        
        return analysis
    
    def _generate_final_codes(self) -> dict[str, Any]:
        """Generate the final set of codes for this coding session."""
        codes = {
            'primary_code': self.stream.current_node.code if self.stream.current_node else None,
            'primary_description': self.stream.current_node.name if self.stream.current_node else None,
            'additional_codes': [],
            'parallel_branch_codes': [],
            'final_suffix_codes': self.stream.final_suffix_codes.copy()
        }
        
        # Add required suffix codes
        for suffix_code in self.stream.required_suffix_codes:
            suffix_node = self.stream.navigator.find_by_code(suffix_code)
            if suffix_node:
                codes['additional_codes'].append({
                    'code': suffix_code,
                    'description': suffix_node.name,
                    'type': 'required_suffix'
                })
        
        # Add parallel branch codes
        for branch_code in self.stream.parallel_branches:
            branch_node = self.stream.navigator.find_by_code(branch_code)
            if branch_node:
                codes['parallel_branch_codes'].append({
                    'code': branch_code,
                    'description': branch_node.name,
                    'type': 'parallel_branch'
                })
        
        return codes
    
    def _validate_clinical_guidelines(self) -> dict[str, Any]:
        """Validate final coding against clinical guidelines."""
        validation = {
            'guideline_compliance': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        if not self.stream.current_node:
            validation['errors'].append('No final code selected')
            validation['guideline_compliance'] = False
            return validation
        
        current_code = self.stream.current_node.code
        
        # Check for common coding issues
        if len(current_code) < 3:
            validation['warnings'].append('Code may be too broad for accurate diagnosis')
        
        # Check notes compliance
        current_notes = getattr(self.stream.current_node, 'notes', {})
        
        # Check excludes1 compliance
        if 'excludes1' in current_notes:
            excluded_codes = [code for code, _ in current_notes['excludes1'] if code]
            for excluded_code in excluded_codes:
                if excluded_code in self.stream.path:
                    validation['errors'].append(f'Code {excluded_code} violates excludes1 rule from {current_code}')
                    validation['guideline_compliance'] = False
        
        # Check codeFirst compliance
        if 'codeFirst' in current_notes:
            required_codes = [code for code, _ in current_notes['codeFirst'] if code]
            if required_codes and not any(code in self.stream.required_prefix_codes for code in required_codes):
                validation['warnings'].append(f'Consider coding {required_codes[0]} first per guidelines')
        
        # Check for missing 7th character when required
        if len(current_code) == 6 and not self.stream.final_suffix_codes:
            validation['suggestions'].append('7th character extension may be required for encounter type')
        
        return validation
    
    def _generate_recommendations(self) -> list[dict[str, Any]]:
        """Generate recommendations for improving coding accuracy."""
        recommendations = []
        
        # Recommend more specific coding if available
        if self.stream.current_node and len(self.stream.current_node.children) > 0:
            recommendations.append({
                'type': 'specificity',
                'priority': 'medium',
                'description': 'More specific codes available',
                'action': f'Consider navigating to one of: {[child.code for child in self.stream.current_node.children[:3]]}',
                'benefit': 'Improved diagnostic specificity and billing accuracy'
            })
        
        # Recommend parallel coding if codeAlso present
        if self.stream.parallel_branches:
            recommendations.append({
                'type': 'parallel_coding',
                'priority': 'high',
                'description': 'Additional codes should be considered',
                'action': f'Consider parallel coding with: {self.stream.parallel_branches[:2]}',
                'benefit': 'Complete clinical picture and guideline compliance'
            })
        
        # Recommend clinical context improvement
        if not self.stream.clinical_context:
            recommendations.append({
                'type': 'clinical_context',
                'priority': 'high',
                'description': 'No clinical context provided',
                'action': 'Provide detailed clinical notes for better coding accuracy',
                'benefit': 'More accurate code selection and better audit trail'
            })
        
        return recommendations