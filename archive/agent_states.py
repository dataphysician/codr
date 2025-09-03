"""
Agent State Management for Clinical Decision Engine
==================================================

This module handles state management for clinical coding trajectories,
including Burr orchestrator integration and state persistence.
"""

import time
from typing import Any
from dataclasses import dataclass


@dataclass
class TrajectoryDecision:
    """Represents a single decision point in the clinical coding trajectory."""
    
    def __init__(self, node_code: str, clinical_context: Any, llm_reasoning: dict, selected_action: dict, timestamp: float | None = None):
        self.node_code = node_code
        self.clinical_context = clinical_context  # Clinical note excerpt that guided decision
        self.llm_reasoning = llm_reasoning  # LLM's structured reasoning
        self.selected_action = selected_action  # The action taken
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'node_code': self.node_code,
            'clinical_context': self.clinical_context,
            'llm_reasoning': self.llm_reasoning,
            'selected_action': self.selected_action,
            'timestamp': self.timestamp
        }


class ClinicalTraversalStream:
    """
    Individual traversal stream with state management and note-driven navigation.
    Maintains ancestor provenance with reasoning chain and enforces clinical coding rules.
    """
    
    def __init__(self, navigator, start_code: str, stream_id: str):
        self.navigator = navigator
        self.stream_id = stream_id
        self.start_code = start_code
        self.current_node = navigator.find_by_code(start_code)
        self.path = [start_code] if self.current_node else []
        self.blocked_codes = set()  # Codes blocked by excludes rules
        self.required_prefix_codes = []  # From codeFirst
        self.required_suffix_codes = []  # From useAdditionalCode
        self.parallel_branches = []  # From codeAlso
        self.final_suffix_codes = []  # From sevenChrNote
        self.excludes2_blockers = {}  # Map of blocking node -> blocked codes
        
        # Enhanced reasoning tracking
        self.reasoning_chain = []  # List of TrajectoryDecision objects
        self.clinical_context = ""  # Current clinical note context
        self.pending_actions = []  # List of pending actions to be performed
        self.action_dependencies = {}  # Map of action dependencies
        
        if not self.current_node:
            raise ValueError(f"Invalid start code: {start_code}")
    
    def set_clinical_context(self, clinical_note: str):
        """Set the clinical context/note for current decision making."""
        self.clinical_context = clinical_note
    
    def get_current_state(self) -> dict[str, Any]:
        """Get complete current state with reasoning chain for LLM decision making."""
        if not self.current_node:
            return None
        
        ancestors = self._get_ancestor_provenance_with_reasoning()
        navigation_options = self._get_navigation_options()
        trajectory_options = self._get_trajectory_decision_options()
        
        return {
            'stream_id': self.stream_id,
            'current_node': {
                'code': self.current_node.code,
                'name': self.current_node.name,
                'element_type': getattr(self.current_node, 'element_type', None),
                'is_leaf': len(self.current_node.children) == 0
            },
            'ancestors_with_reasoning': ancestors,
            'path': self.path.copy(),
            'reasoning_chain': [decision.to_dict() for decision in self.reasoning_chain],
            'clinical_context': self.clinical_context,
            'navigation_options': navigation_options,
            'trajectory_options': trajectory_options,
            'pending_actions': [action for action in self.pending_actions],
            'action_dependencies': self.action_dependencies.copy(),
            'constraints': {
                'blocked_codes': list(self.blocked_codes),
                'required_prefix': self.required_prefix_codes.copy(),
                'required_suffix': self.required_suffix_codes.copy(),
                'parallel_branches': self.parallel_branches.copy(),
                'final_suffix': self.final_suffix_codes.copy()
            },
            'decision_needed': self._needs_decision(),
            'deterministic_next_actions': self._get_deterministic_next_actions()
        }
    
    def _get_ancestor_provenance_with_reasoning(self) -> list[dict[str, Any]]:
        """Get ancestor chain with associated LLM reasoning for each decision."""
        ancestors = []
        if self.current_node and hasattr(self.current_node, 'path'):
            for ancestor in self.current_node.path[1:]:  # Skip root
                ancestor_info = {
                    'code': ancestor.code,
                    'name': ancestor.name,
                    'element_type': getattr(ancestor, 'element_type', None),
                    'reasoning': None,
                    'clinical_citations': []
                }
                
                # Find associated reasoning from decision chain
                for decision in self.reasoning_chain:
                    if decision.node_code == ancestor.code:
                        ancestor_info['reasoning'] = decision.llm_reasoning
                        ancestor_info['clinical_citations'] = decision.clinical_context
                        break
                
                ancestors.append(ancestor_info)
        return ancestors
    
    def _get_navigation_options(self) -> dict[str, Any]:
        """Get available navigation options with validity checks."""
        if not self.current_node:
            return {'children': [], 'jump_codes': [], 'valid_moves': []}
        
        # Direct children
        children = []
        for child in self.current_node.children:
            is_valid = self._is_valid_move(child.code) # NOTE: Every child code of the current node is subjected to move validation
            children.append({
                'code': child.code,
                'name': child.name,
                'element_type': getattr(child, 'element_type', None),
                'valid': is_valid,
                'block_reason': self._get_block_reason(child.code) if not is_valid else None # NOTE: When a child is not valid the block reason is supplied
            })
        
        # Jump codes from notes
        jump_codes = []
        notes = getattr(self.current_node, 'notes', {})
        for note_type, note_list in notes.items():
            for code, description in note_list:
                if code and code != self.current_node.code:
                    is_valid = self._is_valid_move(code)
                    jump_codes.append({
                        'code': code,
                        'description': description,
                        'note_type': note_type,
                        'valid': is_valid,
                        'block_reason': self._get_block_reason(code) if not is_valid else None
                    })
        
        # Valid moves only
        valid_moves = [opt for opt in children + jump_codes if opt['valid']]
        
        return {
            'children': children,
            'jump_codes': jump_codes,
            'valid_moves': valid_moves
        }
    
    def _get_trajectory_decision_options(self) -> dict[str, Any]:
        """Get structured decision options for LLM trajectory planning."""
        if not self.current_node:
            return {}
        
        options = {
            'child_navigation': {
                'description': 'Navigate to more specific child codes',
                'available': len(self.current_node.children) > 0,
                'options': []
            },
            'note_actions': {
                'description': 'Execute note-driven actions (jumps, constraints)',
                'available': bool(self._get_note_actions()),
                'options': []
            },
            'leaf_completion': {
                'description': 'Complete coding at current leaf node',
                'available': len(self.current_node.children) == 0,
                'requirements': self.get_completion_requirements()
            },
            'parallel_branching': {
                'description': 'Create parallel coding streams',
                'available': len(self.parallel_branches) > 0,
                'branch_codes': self.parallel_branches.copy()
            },
            'stop_and_question': {
                'description': 'Request additional clinical information',
                'available': True,
                'reasons': self._get_stop_reasons()
            }
        }
        
        # Populate child navigation options
        if options['child_navigation']['available']:
            for child in self.current_node.children:
                if self._is_valid_move(child.code):
                    options['child_navigation']['options'].append({
                        'code': child.code,
                        'name': child.name,
                        'element_type': getattr(child, 'element_type', None)
                    })
        
        # Populate note action options
        note_actions = self._get_note_actions()
        for note_type, action in note_actions.items():
            if isinstance(action, dict) and 'action' in action:
                options['note_actions']['options'].append({
                    'note_type': note_type,
                    'action': action['action'],
                    'instruction': action.get('instruction', ''),
                    'codes': action.get('blocked_codes', action.get('required_codes', action.get('additional_codes', [])))
                })
        
        return options
    
    def _get_note_actions(self) -> dict[str, Any]:
        """Extract actionable note instructions."""
        if not self.current_node:
            return {}
        
        notes = getattr(self.current_node, 'notes', {})
        actions = {}
        
        for note_type, note_list in notes.items():
            if note_type == 'includes' or note_type == 'inclusionTerm':
                # Context only - no action required
                actions[note_type] = [(code, desc) for code, desc in note_list]
            
            elif note_type == 'excludes1':
                # Block entire path if moving toward these codes
                actions[note_type] = {
                    'instruction': 'Path termination if moving toward these codes',
                    'blocked_codes': [code for code, _ in note_list if code],
                    'action': 'terminate_path'
                }
            
            elif note_type == 'excludes2':
                # Block future moves toward these codes from this point forward
                actions[note_type] = {
                    'instruction': 'Block future moves toward these codes',
                    'blocked_codes': [code for code, _ in note_list if code],
                    'action': 'block_future_moves'
                }
            
            elif note_type == 'codeFirst':
                # Require these codes as prefix before proceeding
                actions[note_type] = {
                    'instruction': 'These codes must be validated before proceeding',
                    'required_codes': [code for code, _ in note_list if code],
                    'action': 'require_prefix'
                }
            
            elif note_type == 'useAdditionalCode':
                # Require these codes alongside current trajectory
                actions[note_type] = {
                    'instruction': 'These codes must accompany the current path',
                    'additional_codes': [code for code, _ in note_list if code],
                    'action': 'require_suffix'
                }
            
            elif note_type == 'codeAlso':
                # Start parallel branches
                actions[note_type] = {
                    'instruction': 'Start parallel traversal branches',
                    'branch_codes': [code for code, _ in note_list if code],
                    'action': 'create_parallel_branches'
                }
            
            elif note_type == 'sevenChrNote':
                # Final suffix codes
                actions[note_type] = {
                    'instruction': 'These codes complete any trajectory from this node',
                    'final_codes': [code for code, _ in note_list if code],
                    'action': 'set_final_suffix'
                }
        
        return actions
    
    def _get_deterministic_next_actions(self) -> list[dict[str, Any]]:
        """Get deterministic actions that will be performed before next LLM decision."""
        actions = []
        
        # Check for required prefix validations
        if self.required_prefix_codes:
            actions.append({
                'type': 'prefix_validation',
                'codes': self.required_prefix_codes,
                'description': f"Validate required prefix codes: {', '.join(self.required_prefix_codes)}"
            })
        
        # Check for automatic constraint enforcement
        current_notes = getattr(self.current_node, 'notes', {})
        if 'excludes1' in current_notes:
            blocked = [code for code, _ in current_notes['excludes1'] if code]
            if blocked:
                actions.append({
                    'type': 'excludes1_enforcement',
                    'blocked_codes': blocked,
                    'description': f"Block codes due to excludes1: {', '.join(blocked[:3])}{'...' if len(blocked) > 3 else ''}"
                })
        
        if 'sevenChrNote' in current_notes:
            final_codes = [code for code, _ in current_notes['sevenChrNote'] if code]
            if final_codes:
                actions.append({
                    'type': 'final_suffix_available',
                    'codes': final_codes,
                    'description': f"Final suffix codes available: {', '.join(final_codes[:3])}{'...' if len(final_codes) > 3 else ''}"
                })
        
        return actions
    
    def _get_stop_reasons(self) -> list[str]:
        """Get reasons why LLM might need to stop and ask questions."""
        reasons = []
        
        # No valid moves available
        nav_options = self._get_navigation_options()
        if len(nav_options['valid_moves']) == 0 and len(self.current_node.children) > 0:
            reasons.append("All child navigation options are blocked by constraints")
        
        # Conflicting constraints
        if self.blocked_codes and self.required_prefix_codes:
            overlap = set(self.blocked_codes) & set(self.required_prefix_codes)
            if overlap:
                reasons.append(f"Conflicting constraints: codes {list(overlap)} are both required and blocked")
        
        # Missing clinical context
        if not self.clinical_context:
            reasons.append("No clinical context provided for decision making")
        
        # Multiple equally valid options
        if len(nav_options['valid_moves']) > 3:
            reasons.append(f"Too many valid options ({len(nav_options['valid_moves'])}) - need clinical guidance")
        
        return reasons
    
    def _is_valid_move(self, target_code: str) -> bool:
        """Check if move to target code is valid given current constraints."""
        # Check excludes1 blocks (from current node and ancestors)
        if target_code in self.blocked_codes:
            return False
        
        # Check excludes2 blocks from ancestor nodes
        for blocking_node, blocked_set in self.excludes2_blockers.items():
            if target_code in blocked_set:
                return False
        
        # Check if code exists
        target_node = self.navigator.find_by_code(target_code)
        if not target_node:
            return False
        
        return True
    
    def _get_block_reason(self, target_code: str) -> str:
        """Get reason why a move is blocked."""
        if target_code in self.blocked_codes:
            return "Blocked by excludes1 rule"
        
        for blocking_node, blocked_set in self.excludes2_blockers.items():
            if target_code in blocked_set:
                return f"Blocked by excludes2 rule from {blocking_node}"
        
        target_node = self.navigator.find_by_code(target_code)
        if not target_node:
            return "Code does not exist"
        
        return "Unknown block reason"
    
    def _needs_decision(self) -> bool:
        """Determine if external decision (LLM/user) is needed."""
        if not self.current_node:
            return False
        
        # If it's a leaf and we have valid suffix codes, we're done
        if len(self.current_node.children) == 0:
            return len(self.final_suffix_codes) == 0  # Need decision if no final codes
        
        # If we have valid moves, we need a decision
        nav_options = self._get_navigation_options()
        return len(nav_options['valid_moves']) > 0
    
    def navigate_to(self, target_code: str, validate_constraints: bool = True) -> bool:
        """Navigate to target code with constraint validation."""
        if not self._is_valid_move(target_code):
            if validate_constraints:
                raise ValueError(f"Invalid move to {target_code}: {self._get_block_reason(target_code)}")
            else:
                return False
        
        target_node = self.navigator.find_by_code(target_code)
        if not target_node:
            raise ValueError(f"Target code {target_code} not found")
        
        # Update current state
        self.current_node = target_node
        self.path.append(target_code)
        
        # Apply note constraints from new current node
        self._apply_note_constraints()
        
        return True
    
    def _apply_note_constraints(self):
        """Apply note-driven constraints from current node."""
        if not self.current_node:
            return
        
        notes = getattr(self.current_node, 'notes', {})
        
        # Process excludes1 - block these codes immediately
        if 'excludes1' in notes:
            for code, _ in notes['excludes1']:
                if code:
                    self.blocked_codes.add(code)
        
        # Process excludes2 - block future moves to these codes
        if 'excludes2' in notes:
            blocked_codes = {code for code, _ in notes['excludes2'] if code}
            if blocked_codes:
                self.excludes2_blockers[self.current_node.code] = blocked_codes
        
        # Process codeFirst - add to required prefix
        if 'codeFirst' in notes:
            prefix_codes = [code for code, _ in notes['codeFirst'] if code]
            self.required_prefix_codes.extend(prefix_codes)
        
        # Process useAdditionalCode - add to required suffix
        if 'useAdditionalCode' in notes:
            suffix_codes = [code for code, _ in notes['useAdditionalCode'] if code]
            self.required_suffix_codes.extend(suffix_codes)
        
        # Process codeAlso - create parallel branches
        if 'codeAlso' in notes:
            branch_codes = [code for code, _ in notes['codeAlso'] if code]
            self.parallel_branches.extend(branch_codes)
        
        # Process sevenChrNote - set final suffix codes
        if 'sevenChrNote' in notes:
            final_codes = [code for code, _ in notes['sevenChrNote'] if code]
            self.final_suffix_codes.extend(final_codes)
    
    def get_completion_requirements(self) -> dict[str, Any]:
        """Get what's needed to complete this traversal stream."""
        requirements = {
            'prefix_codes_needed': self.required_prefix_codes.copy(),
            'suffix_codes_needed': self.required_suffix_codes.copy(),
            'final_codes_available': self.final_suffix_codes.copy(),
            'parallel_branches_available': self.parallel_branches.copy(),
            'is_complete': self._is_complete()
        }
        return requirements
    
    def _is_complete(self) -> bool:
        """Check if traversal stream is complete."""
        # Complete if we're at a leaf with final suffix codes or no requirements
        if not self.current_node:
            return False
        
        is_leaf = len(self.current_node.children) == 0
        has_final_codes = len(self.final_suffix_codes) > 0
        no_requirements = (len(self.required_prefix_codes) == 0 and 
                          len(self.required_suffix_codes) == 0)
        
        return is_leaf and (has_final_codes or no_requirements)
    
    def get_decision_summary(self) -> dict[str, Any]:
        """Get a summary of all decisions made in this stream."""
        return {
            'stream_id': self.stream_id,
            'start_code': self.start_code,
            'current_code': self.current_node.code if self.current_node else None,
            'path_length': len(self.path),
            'decisions_made': len(self.reasoning_chain),
            'clinical_context_set': bool(self.clinical_context),
            'pending_actions': len(self.pending_actions),
            'completion_status': self._is_complete(),
            'constraint_counts': {
                'blocked_codes': len(self.blocked_codes),
                'required_prefix': len(self.required_prefix_codes),
                'required_suffix': len(self.required_suffix_codes),
                'parallel_branches': len(self.parallel_branches),
                'final_suffix': len(self.final_suffix_codes)
            }
        }
    
    # Dictionary-based state conversion methods
    def to_dict_by_codes(self) -> dict[str, Any]:
        """Convert stream state to dictionary with node codes as keys."""
        if not self.current_node:
            return {}
        
        # Create code-keyed dictionary
        code_dict = {}
        
        # Add all nodes in current path with their states
        for node_code in self.path:
            node = self.navigator.find_by_code(node_code)
            if node:
                # Find reasoning for this node
                reasoning = None
                clinical_citations = []
                for decision in self.reasoning_chain:
                    if decision.node_code == node_code:
                        reasoning = decision.llm_reasoning
                        clinical_citations = decision.clinical_context
                        break
                
                # Check if this is current node
                is_current = node_code == self.current_node.code
                
                code_dict[node_code] = {
                    'name': node.name,
                    'element_type': getattr(node, 'element_type', None),
                    'is_current': is_current,
                    'is_leaf': len(node.children) == 0,
                    'reasoning': reasoning,
                    'clinical_citations': clinical_citations,
                    'notes': getattr(node, 'notes', {}),
                    'children_codes': [child.code for child in node.children],
                    'position_in_path': self.path.index(node_code)
                }
                
                # Add constraints specific to this node
                if is_current:
                    code_dict[node_code]['active_constraints'] = {
                        'blocked_codes': list(self.blocked_codes),
                        'required_prefix': self.required_prefix_codes.copy(),
                        'required_suffix': self.required_suffix_codes.copy(),
                        'parallel_branches': self.parallel_branches.copy(),
                        'final_suffix': self.final_suffix_codes.copy()
                    }
        
        # Add pending actions keyed by target code
        pending_by_code = {}
        for action in self.pending_actions:
            target_code = action['target_code']
            if target_code not in pending_by_code:
                pending_by_code[target_code] = []
            pending_by_code[target_code].append(action)
        
        # Add blocked codes with reasons
        blocked_codes_dict = {}
        for blocked_code in self.blocked_codes:
            blocked_codes_dict[blocked_code] = {
                'reason': 'excludes1_rule',
                'blocking_nodes': []
            }
            # Find which nodes caused the block
            for node_code in self.path:
                node = self.navigator.find_by_code(node_code)
                if node and hasattr(node, 'notes'):
                    notes = getattr(node, 'notes', {})
                    if 'excludes1' in notes:
                        for code, _ in notes['excludes1']:
                            if code == blocked_code:
                                blocked_codes_dict[blocked_code]['blocking_nodes'].append(node_code)
        
        return {
            'stream_metadata': {
                'stream_id': self.stream_id,
                'start_code': self.start_code,
                'current_code': self.current_node.code,
                'path': self.path,
                'clinical_context': self.clinical_context
            },
            'nodes_by_code': code_dict,
            'pending_actions_by_code': pending_by_code,
            'blocked_codes_by_code': blocked_codes_dict,
            'completion_status': {
                'is_complete': self._is_complete(),
                'requirements': self.get_completion_requirements()
            }
        }
    
    @classmethod
    def from_dict_by_codes(cls, navigator, code_dict_data: dict[str, Any]):
        """Reconstruct stream from dictionary with node codes as keys."""
        metadata = code_dict_data['stream_metadata']
        
        # Create new stream
        stream = cls(navigator, metadata['start_code'], metadata['stream_id'])
        
        # Restore path and current node
        stream.path = metadata['path']
        stream.current_node = navigator.find_by_code(metadata['current_code'])
        stream.clinical_context = metadata['clinical_context']
        
        # Restore reasoning chain
        nodes_data = code_dict_data['nodes_by_code']
        for node_code in stream.path:
            node_data = nodes_data.get(node_code, {})
            if node_data.get('reasoning'):
                decision = TrajectoryDecision(
                    node_code=node_code,
                    clinical_context=node_data.get('clinical_citations', []),
                    llm_reasoning=node_data['reasoning'],
                    selected_action={'type': 'navigate_child', 'target_code': node_code}
                )
                stream.reasoning_chain.append(decision)
        
        # Restore constraints from current node
        current_node_data = nodes_data.get(metadata['current_code'], {})
        constraints = current_node_data.get('active_constraints', {})
        stream.blocked_codes = set(constraints.get('blocked_codes', []))
        stream.required_prefix_codes = constraints.get('required_prefix', [])
        stream.required_suffix_codes = constraints.get('required_suffix', [])
        stream.parallel_branches = constraints.get('parallel_branches', [])
        stream.final_suffix_codes = constraints.get('final_suffix', [])
        
        # Restore pending actions
        pending_actions_data = code_dict_data.get('pending_actions_by_code', {})
        for target_code, actions_list in pending_actions_data.items():
            stream.pending_actions.extend(actions_list)
        
        return stream
    
    # Burr orchestrator integration methods
    def to_burr_state(self) -> dict[str, Any]:
        """
        Convert stream state to Burr-compatible format for orchestration.
        Burr expects flat key-value state dictionaries.
        """
        state = {
            # Core stream identity
            'stream_id': self.stream_id,
            'start_code': self.start_code,
            'current_code': self.current_node.code if self.current_node else None,
            'current_name': self.current_node.name if self.current_node else None,
            
            # Path and reasoning
            'path': self.path,
            'path_length': len(self.path),
            'clinical_context': self.clinical_context,
            
            # Current state flags
            'is_complete': self._is_complete(),
            'is_leaf': len(self.current_node.children) == 0 if self.current_node else False,
            'decision_needed': self._needs_decision(),
            
            # Action counts
            'pending_actions_count': len(self.pending_actions),
            'decisions_made': len(self.reasoning_chain),
            
            # Constraints (as counts for Burr state)
            'blocked_codes_count': len(self.blocked_codes),
            'required_prefix_count': len(self.required_prefix_codes),
            'required_suffix_count': len(self.required_suffix_codes),
            'parallel_branches_count': len(self.parallel_branches),
            'final_suffix_count': len(self.final_suffix_codes),
            
            # Latest reasoning (for Burr decision tracking)
            'last_llm_reasoning': None,
            'last_clinical_citations': [],
            'last_action_type': None
        }
        
        # Add latest reasoning if available
        if self.reasoning_chain:
            latest_decision = self.reasoning_chain[-1]
            state['last_llm_reasoning'] = latest_decision.llm_reasoning
            state['last_clinical_citations'] = latest_decision.clinical_context
            if hasattr(latest_decision, 'selected_action'):
                state['last_action_type'] = latest_decision.selected_action.get('type', None)
        
        # Add node codes as separate keys for Burr access
        for i, node_code in enumerate(self.path):
            state[f'node_{i}_code'] = node_code
            node = self.navigator.find_by_code(node_code)
            if node:
                state[f'node_{i}_name'] = node.name
                state[f'node_{i}_element_type'] = getattr(node, 'element_type', None)
        
        # Add pending action details as flat keys
        for i, action in enumerate(self.pending_actions):
            prefix = f'pending_action_{i}'
            state[f'{prefix}_id'] = action['action_id']
            state[f'{prefix}_type'] = action['type']
            state[f'{prefix}_target_code'] = action['target_code']
            state[f'{prefix}_priority'] = action['priority']
            state[f'{prefix}_has_dependencies'] = len(action.get('dependencies', [])) > 0
        
        return state
    
    @classmethod
    def from_burr_state(cls, navigator, burr_state: dict[str, Any]):
        """Reconstruct stream from Burr state format."""
        # Create stream with basic info
        stream = cls(navigator, burr_state['start_code'], burr_state['stream_id'])
        
        # Restore path
        stream.path = burr_state['path']
        stream.current_node = navigator.find_by_code(burr_state['current_code'])
        stream.clinical_context = burr_state['clinical_context']
        
        # Note: Full constraint restoration would require additional state
        # This is a minimal reconstruction for Burr orchestration
        
        return stream


class ClinicalDecisionEngine:
    """
    Clinical decision engine managing multiple traversal streams.
    Handles stream lifecycle and coordination.
    """
    
    def __init__(self, navigator):
        self.navigator = navigator
        self.active_streams = {}  # Track multiple parallel traversal streams
    
    def create_traversal_stream(self, start_code: str, stream_id: str | None = None) -> ClinicalTraversalStream:
        """Create a new traversal stream starting from given code."""
        if stream_id is None:
            stream_id = f"stream_{len(self.active_streams)}"
        
        stream = ClinicalTraversalStream(
            navigator=self.navigator,
            start_code=start_code,
            stream_id=stream_id
        )
        self.active_streams[stream_id] = stream
        return stream
    
    def get_stream(self, stream_id: str) -> ClinicalTraversalStream | None:
        """Get existing traversal stream by ID."""
        return self.active_streams.get(stream_id)
    
    def close_stream(self, stream_id: str):
        """Close and remove a traversal stream."""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
    
    def get_all_stream_states(self) -> dict[str, dict[str, Any]]:
        """Get current state of all active streams."""
        return {
            stream_id: stream.get_current_state() 
            for stream_id, stream in self.active_streams.items()
        }
    
    def get_burr_state_summary(self) -> dict[str, Any]:
        """Get Burr-compatible summary of all streams."""
        return {
            'active_streams_count': len(self.active_streams),
            'stream_ids': list(self.active_streams.keys()),
            'streams_needing_decision': [
                stream_id for stream_id, stream in self.active_streams.items()
                if stream._needs_decision()
            ],
            'completed_streams': [
                stream_id for stream_id, stream in self.active_streams.items()
                if stream._is_complete()
            ]
        }