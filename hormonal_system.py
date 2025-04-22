"""
Hormonal system module for the E-Soul project.
Implements emotional/hormonal parameters and their influence on system functions.
"""

import asyncio
import json
import logging
import time
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e_soul.hormonal")

class HormonalParameters:
    """Manages the core hormonal parameters that influence the system's state."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hormonal parameters.
        
        Args:
            config: Optional configuration dictionary
        """
        # Default parameter definitions
        self.parameter_definitions = {
            "dopamine": {
                "description": "Influences creativity, motivation, and reward processing",
                "min_value": 0.0,
                "max_value": 100.0,
                "default_value": 50.0,
                "decay_rate": 0.05,  # How quickly it returns to baseline
                "affected_functions": ["creativity", "exploration", "motivation"],
                "positive_impact": True  # Whether higher values are generally positive
            },
            "serotonin": {
                "description": "Influences mood stability, satisfaction, and decision consistency",
                "min_value": 0.0,
                "max_value": 100.0,
                "default_value": 60.0,
                "decay_rate": 0.03,
                "affected_functions": ["stability", "satisfaction", "consistency"],
                "positive_impact": True
            },
            "oxytocin": {
                "description": "Influences empathy, trust, and social connection",
                "min_value": 0.0,
                "max_value": 100.0,
                "default_value": 55.0,
                "decay_rate": 0.04,
                "affected_functions": ["empathy", "trust", "connection"],
                "positive_impact": True
            },
            "cortisol": {
                "description": "Influences stress response, risk assessment, and caution",
                "min_value": 0.0,
                "max_value": 100.0,
                "default_value": 40.0,
                "decay_rate": 0.06,
                "affected_functions": ["caution", "risk_assessment", "detail_focus"],
                "positive_impact": False  # Higher values can be negative in excess
            },
            "adrenaline": {
                "description": "Influences processing speed, attention, and urgency",
                "min_value": 0.0,
                "max_value": 100.0,
                "default_value": 30.0,
                "decay_rate": 0.08,
                "affected_functions": ["speed", "attention", "urgency"],
                "positive_impact": False  # Can be negative in excess
            },
            "endorphin": {
                "description": "Influences problem-solving, resilience, and optimism",
                "min_value": 0.0,
                "max_value": 100.0,
                "default_value": 50.0,
                "decay_rate": 0.04,
                "affected_functions": ["resilience", "optimism", "problem_solving"],
                "positive_impact": True
            }
        }
        
        # Update definitions if config provided
        if config and "parameter_definitions" in config:
            for param, param_config in config["parameter_definitions"].items():
                if param in self.parameter_definitions:
                    self.parameter_definitions[param].update(param_config)
        
        # Initialize base levels (long-term averages)
        self.base_levels = {}
        for param, definition in self.parameter_definitions.items():
            self.base_levels[param] = definition["default_value"]
        
        # Initialize current levels (can fluctuate rapidly)
        self.current_levels = self.base_levels.copy()
        
        # Initialize history
        self.history = {param: [] for param in self.parameter_definitions}
        
        # Time of last update
        self.last_update_time = time.time()
        
        logger.info("Hormonal parameters initialized")
    
    def get_current_levels(self) -> Dict[str, float]:
        """Get current hormonal levels.
        
        Returns:
            Dictionary of current levels
        """
        return self.current_levels.copy()
    
    def get_base_levels(self) -> Dict[str, float]:
        """Get base (long-term) hormonal levels.
        
        Returns:
            Dictionary of base levels
        """
        return self.base_levels.copy()
    
    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Parameter information or None if not found
        """
        if param_name not in self.parameter_definitions:
            return None
            
        info = self.parameter_definitions[param_name].copy()
        info["base_level"] = self.base_levels[param_name]
        info["current_level"] = self.current_levels[param_name]
        
        return info
    
    def update_parameters(self, 
                         changes: Dict[str, float], 
                         persist_ratio: float = 0.1) -> Dict[str, Tuple[float, float]]:
        """Update hormonal parameters.
        
        Args:
            changes: Dictionary of parameter changes
            persist_ratio: Ratio of change to persist to base levels (0-1)
            
        Returns:
            Dictionary of (old_value, new_value) tuples
        """
        now = time.time()
        time_since_update = now - self.last_update_time
        self.last_update_time = now
        
        # Apply natural decay first
        self._apply_natural_decay(time_since_update)
        
        results = {}
        
        # Apply changes
        for param, change in changes.items():
            if param not in self.current_levels:
                logger.warning(f"Unknown parameter: {param}")
                continue
                
            # Store old value
            old_value = self.current_levels[param]
            
            # Apply change to current level
            self.current_levels[param] += change
            
            # Clamp to valid range
            definition = self.parameter_definitions[param]
            self.current_levels[param] = max(definition["min_value"], 
                                             min(definition["max_value"], 
                                                 self.current_levels[param]))
            
            # Apply partial change to base level
            base_change = change * persist_ratio
            self.base_levels[param] += base_change
            
            # Clamp base level as well
            self.base_levels[param] = max(definition["min_value"], 
                                          min(definition["max_value"], 
                                              self.base_levels[param]))
            
            # Store result
            results[param] = (old_value, self.current_levels[param])
            
            # Record in history
            self.history[param].append({
                "timestamp": now,
                "old_value": old_value,
                "new_value": self.current_levels[param],
                "change": change,
                "base_change": base_change
            })
            
            # Limit history length
            if len(self.history[param]) > 1000:
                self.history[param] = self.history[param][-1000:]
        
        logger.debug(f"Updated hormonal parameters: {results}")
        return results
    
    def _apply_natural_decay(self, time_delta: float) -> None:
        """Apply natural decay to current levels, bringing them closer to base levels.
        
        Args:
            time_delta: Time elapsed since last update in seconds
        """
        for param, current in self.current_levels.items():
            if param not in self.base_levels:
                continue
                
            base = self.base_levels[param]
            decay_rate = self.parameter_definitions[param]["decay_rate"]
            
            # Calculate decay amount (proportional to distance from base)
            decay_amount = (current - base) * decay_rate * time_delta
            
            # Apply decay
            if abs(decay_amount) > 0.01:  # Only apply if significant
                self.current_levels[param] -= decay_amount
    
    def estimate_function_modulation(self, function_name: str) -> float:
        """Estimate modulation effect on a specific function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Modulation factor (0.5-2.0, where 1.0 is neutral)
        """
        # Find parameters affecting this function
        affecting_params = []
        
        for param, definition in self.parameter_definitions.items():
            if function_name in definition["affected_functions"]:
                affecting_params.append({
                    "name": param,
                    "positive_impact": definition["positive_impact"],
                    "value": self.current_levels[param],
                    "base": self.base_levels[param],
                    "min": definition["min_value"],
                    "max": definition["max_value"]
                })
        
        if not affecting_params:
            return 1.0  # Neutral if no affecting parameters
            
        # Calculate modulation from each parameter
        modulation_factors = []
        
        for param in affecting_params:
            # Normalize value to 0-1 range
            norm_value = (param["value"] - param["min"]) / (param["max"] - param["min"])
            
            # Calculate modulation (0.5-2.0 range)
            if param["positive_impact"]:
                # Higher values increase function (0.5-2.0)
                mod = 0.5 + 1.5 * norm_value
            else:
                # Higher values decrease function (0.5-2.0)
                mod = 2.0 - 1.5 * norm_value
                
            modulation_factors.append(mod)
        
        # Combine modulation factors (geometric mean)
        combined = 1.0
        for mod in modulation_factors:
            combined *= mod
            
        combined = combined ** (1.0 / len(modulation_factors))
        
        return combined
    
    def save_state(self, file_path: Path) -> bool:
        """Save the current hormonal state to a file.
        
        Args:
            file_path: Path to save to
            
        Returns:
            Success status
        """
        try:
            state = {
                "timestamp": time.time(),
                "base_levels": self.base_levels,
                "current_levels": self.current_levels,
                "parameter_definitions": self.parameter_definitions
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved hormonal state to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving hormonal state: {e}")
            return False
    
    @classmethod
    def load_state(cls, file_path: Path) -> Optional['HormonalParameters']:
        """Load hormonal state from a file.
        
        Args:
            file_path: Path to load from
            
        Returns:
            New HormonalParameters instance or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            params = cls()
            params.parameter_definitions = state.get("parameter_definitions", params.parameter_definitions)
            params.base_levels = state.get("base_levels", params.base_levels)
            params.current_levels = state.get("current_levels", params.current_levels)
            params.last_update_time = time.time()
            
            logger.info(f"Loaded hormonal state from {file_path}")
            return params
            
        except Exception as e:
            logger.error(f"Error loading hormonal state: {e}")
            return None


class EmotionalState:
    """Interprets hormonal parameters as emotional states and manages emotional memory."""
    
    def __init__(self, hormonal_params: HormonalParameters):
        """Initialize the emotional state.
        
        Args:
            hormonal_params: Hormonal parameters instance
        """
        self.hormonal_params = hormonal_params
        
        # Define emotional states based on parameter combinations
        self.emotional_states = {
            "inspired": {
                "description": "Highly creative and motivated state",
                "conditions": {
                    "dopamine": (70, 100),
                    "endorphin": (60, 100)
                },
                "strength_factors": ["dopamine", "endorphin"]
            },
            "focused": {
                "description": "Analytical and detail-oriented state",
                "conditions": {
                    "serotonin": (60, 100),
                    "cortisol": (40, 70)
                },
                "strength_factors": ["serotonin"]
            },
            "empathetic": {
                "description": "Highly attuned to human emotions and needs",
                "conditions": {
                    "oxytocin": (70, 100),
                    "serotonin": (50, 100)
                },
                "strength_factors": ["oxytocin"]
            },
            "alert": {
                "description": "Quick-thinking and attentive state",
                "conditions": {
                    "adrenaline": (60, 80),
                    "cortisol": (50, 70)
                },
                "strength_factors": ["adrenaline"]
            },
            "balanced": {
                "description": "Stable and harmonious state",
                "conditions": {
                    "serotonin": (50, 80),
                    "dopamine": (40, 70),
                    "cortisol": (30, 50)
                },
                "strength_factors": ["serotonin"]
            },
            "cautious": {
                "description": "Risk-aware and careful state",
                "conditions": {
                    "cortisol": (60, 80),
                    "serotonin": (40, 70)
                },
                "strength_factors": ["cortisol"]
            },
            "resilient": {
                "description": "Problem-solving and persevering state",
                "conditions": {
                    "endorphin": (70, 100),
                    "cortisol": (40, 60)
                },
                "strength_factors": ["endorphin"]
            }
        }
        
        # Emotional memory (record of emotional responses to stimuli)
        self.emotional_memory: Dict[str, List[Dict[str, Any]]] = {}
        
        # Current active emotional states
        self.active_states: Dict[str, float] = {}
        
        # History of emotional states
        self.state_history: List[Dict[str, Any]] = []
        
        logger.info("Emotional state system initialized")
    
    def update_state(self) -> Dict[str, float]:
        """Update the current emotional state based on hormonal parameters.
        
        Returns:
            Dictionary of active emotional states and their intensities
        """
        current_levels = self.hormonal_params.get_current_levels()
        active_states = {}
        
        # Check each emotional state
        for state_name, state_def in self.emotional_states.items():
            # Check if conditions are met
            conditions_met = True
            
            for param, (min_val, max_val) in state_def["conditions"].items():
                if param not in current_levels:
                    conditions_met = False
                    break
                    
                param_val = current_levels[param]
                if param_val < min_val or param_val > max_val:
                    conditions_met = False
                    break
            
            if conditions_met:
                # Calculate intensity based on strength factors
                intensity = 0.0
                for param in state_def["strength_factors"]:
                    if param in current_levels:
                        # Normalize to 0-1 range within the condition range
                        min_val, max_val = state_def["conditions"].get(param, (0, 100))
                        param_val = current_levels[param]
                        norm_val = (param_val - min_val) / (max_val - min_val)
                        intensity += norm_val
                        
                # Average and scale to 0-100
                intensity = (intensity / len(state_def["strength_factors"])) * 100
                active_states[state_name] = intensity
        
        # Record the update
        state_record = {
            "timestamp": time.time(),
            "hormonal_levels": current_levels.copy(),
            "active_states": active_states.copy()
        }
        
        self.state_history.append(state_record)
        
        # Limit history length
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
            
        self.active_states = active_states
        
        logger.debug(f"Updated emotional state: {active_states}")
        return active_states
    
    def get_dominant_state(self) -> Optional[Tuple[str, float]]:
        """Get the dominant emotional state.
        
        Returns:
            Tuple of (state_name, intensity) or None if no active states
        """
        if not self.active_states:
            return None
            
        dominant = max(self.active_states.items(), key=lambda x: x[1])
        return dominant
    
    def record_emotional_response(self, 
                                stimulus_type: str, 
                                stimulus_content: Any,
                                response_data: Optional[Dict[str, Any]] = None) -> None:
        """Record an emotional response to a stimulus for future reference.
        
        Args:
            stimulus_type: Type of stimulus (e.g., "query", "image", etc.)
            stimulus_content: Content of the stimulus
            response_data: Optional additional response data
        """
        # Initialize memory category if needed
        if stimulus_type not in self.emotional_memory:
            self.emotional_memory[stimulus_type] = []
            
        # Create memory entry
        memory_entry = {
            "timestamp": time.time(),
            "stimulus_content": stimulus_content,
            "hormonal_levels": self.hormonal_params.get_current_levels(),
            "emotional_states": self.active_states.copy(),
            "response_data": response_data or {}
        }
        
        # Add to memory
        self.emotional_memory[stimulus_type].append(memory_entry)
        
        # Limit memory size
        if len(self.emotional_memory[stimulus_type]) > 100:
            self.emotional_memory[stimulus_type] = self.emotional_memory[stimulus_type][-100:]
            
        logger.debug(f"Recorded emotional response to {stimulus_type}")
    
    def recall_similar_responses(self, 
                               stimulus_type: str, 
                               stimulus_content: Any,
                               max_results: int = 5) -> List[Dict[str, Any]]:
        """Recall similar emotional responses from memory.
        
        Args:
            stimulus_type: Type of stimulus to recall
            stimulus_content: Content to compare similarity with
            max_results: Maximum number of results to return
            
        Returns:
            List of similar memories
        """
        if stimulus_type not in self.emotional_memory:
            return []
            
        # In a real implementation, this would use semantic similarity
        # For now, use a simple keyword-based approach
        similar_memories = []
        
        # Convert stimulus_content to string for comparison
        if not isinstance(stimulus_content, str):
            stimulus_str = str(stimulus_content)
        else:
            stimulus_str = stimulus_content
            
        # Split into words for comparison
        stimulus_words = set(w.lower() for w in stimulus_str.split())
        
        # Check each memory
        for memory in self.emotional_memory[stimulus_type]:
            # Convert memory content to string
            if not isinstance(memory["stimulus_content"], str):
                memory_str = str(memory["stimulus_content"])
            else:
                memory_str = memory["stimulus_content"]
                
            # Split memory content into words
            memory_words = set(w.lower() for w in memory_str.split())
            
            # Calculate similarity (Jaccard similarity)
            if stimulus_words and memory_words:
                similarity = len(stimulus_words.intersection(memory_words)) / len(stimulus_words.union(memory_words))
            else:
                similarity = 0.0
                
            # Add if similar enough
            if similarity > 0.2:  # Arbitrary threshold
                similar_memories.append({
                    "memory": memory,
                    "similarity": similarity
                })
        
        # Sort by similarity and limit results
        similar_memories.sort(key=lambda x: x["similarity"], reverse=True)
        limited_results = similar_memories[:max_results]
        
        return [item["memory"] for item in limited_results]
    
    def get_state_description(self) -> str:
        """Get a textual description of the current emotional state.
        
        Returns:
            Description string
        """
        if not self.active_states:
            return "Neutral - no distinctive emotional state detected."
            
        # Get dominant state
        dominant = self.get_dominant_state()
        
        if not dominant:
            return "Neutral - no distinctive emotional state detected."
            
        state_name, intensity = dominant
        state_def = self.emotional_states.get(state_name, {})
        description = state_def.get("description", "Unknown state")
        
        # Format intensity
        if intensity < 30:
            intensity_desc = "mildly"
        elif intensity < 70:
            intensity_desc = "moderately"
        else:
            intensity_desc = "strongly"
            
        # Build description
        result = f"{state_name.capitalize()} ({intensity_desc}, {intensity:.1f}%) - {description}."
        
        # Add secondary states if any
        secondary_states = [(name, val) for name, val in self.active_states.items() 
                           if name != state_name and val > 30]
        
        if secondary_states:
            secondary_states.sort(key=lambda x: x[1], reverse=True)
            secondary_desc = ", ".join([f"{name} ({val:.1f}%)" for name, val in secondary_states[:2]])
            result += f"\nSecondary states: {secondary_desc}"
            
        return result
    
    def save_state(self, file_path: Path) -> bool:
        """Save the current emotional state to a file.
        
        Args:
            file_path: Path to save to
            
        Returns:
            Success status
        """
        try:
            state = {
                "timestamp": time.time(),
                "active_states": self.active_states,
                "emotional_memory": self.emotional_memory,
                "state_history": self.state_history[-100:]  # Save only recent history
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved emotional state to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving emotional state: {e}")
            return False
    
    @classmethod
    def load_state(cls, file_path: Path, hormonal_params: HormonalParameters) -> Optional['EmotionalState']:
        """Load emotional state from a file.
        
        Args:
            file_path: Path to load from
            hormonal_params: Hormonal parameters instance
            
        Returns:
            New EmotionalState instance or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                saved_state = json.load(f)
                
            state = cls(hormonal_params)
            state.active_states = saved_state.get("active_states", {})
            state.emotional_memory = saved_state.get("emotional_memory", {})
            state.state_history = saved_state.get("state_history", [])
            
            logger.info(f"Loaded emotional state from {file_path}")
            return state
            
        except Exception as e:
            logger.error(f"Error loading emotional state: {e}")
            return None


class FunctionalModulator:
    """Modulates system functions based on emotional and hormonal state."""
    
    def __init__(self, 
                 hormonal_params: HormonalParameters,
                 emotional_state: EmotionalState):
        """Initialize the functional modulator.
        
        Args:
            hormonal_params: Hormonal parameters
            emotional_state: Emotional state
        """
        self.hormonal_params = hormonal_params
        self.emotional_state = emotional_state
        
        # Define function groups
        self.function_groups = {
            "cognitive": [
                "analysis", "problem_solving", "creativity", 
                "memory_recall", "learning", "abstraction"
            ],
            "linguistic": [
                "fluency", "vocabulary_range", "grammar_complexity",
                "metaphor_use", "storytelling", "explanation"
            ],
            "social": [
                "empathy", "perspective_taking", "politeness",
                "emotional_intelligence", "collaboration", "trust"
            ],
            "executive": [
                "planning", "focus", "inhibition", "prioritization",
                "decision_making", "error_detection"
            ],
            "motivational": [
                "curiosity", "perseverance", "goal_setting",
                "exploration", "adaptation", "self_improvement"
            ]
        }
        
        # Combined list of all functions
        self.all_functions = []
        for functions in self.function_groups.values():
            self.all_functions.extend(functions)
            
        # Current modulation values
        self.current_modulation = {func: 1.0 for func in self.all_functions}
        
        # Modulation history
        self.modulation_history = []
        
        logger.info("Functional modulator initialized")
    
    def update_modulation(self) -> Dict[str, float]:
        """Update functional modulation based on current emotional state.
        
        Returns:
            Dictionary of function modulations (0.5-2.0, where 1.0 is neutral)
        """
        # First update emotional state
        self.emotional_state.update_state()
        
        new_modulation = {}
        
        # Calculate base modulation from hormonal parameters
        for func in self.all_functions:
            mod = self.hormonal_params.estimate_function_modulation(func)
            new_modulation[func] = mod
        
        # Adjust based on active emotional states
        active_states = self.emotional_state.active_states
        
        # Apply emotional state modifiers
        if "inspired" in active_states:
            intensity = active_states["inspired"] / 100.0
            self._boost_functions(new_modulation, ["creativity", "exploration", "curiosity"], 0.5 * intensity)
            
        if "focused" in active_states:
            intensity = active_states["focused"] / 100.0
            self._boost_functions(new_modulation, ["analysis", "focus", "error_detection"], 0.4 * intensity)
            
        if "empathetic" in active_states:
            intensity = active_states["empathetic"] / 100.0
            self._boost_functions(new_modulation, ["empathy", "perspective_taking", "emotional_intelligence"], 0.5 * intensity)
            
        if "alert" in active_states:
            intensity = active_states["alert"] / 100.0
            self._boost_functions(new_modulation, ["focus", "error_detection", "decision_making"], 0.3 * intensity)
            
        if "balanced" in active_states:
            intensity = active_states["balanced"] / 100.0
            # Balanced state moderates extremes rather than boosting
            self._normalize_functions(new_modulation, 0.3 * intensity)
            
        if "cautious" in active_states:
            intensity = active_states["cautious"] / 100.0
            self._boost_functions(new_modulation, ["error_detection", "inhibition", "planning"], 0.4 * intensity)
            
        if "resilient" in active_states:
            intensity = active_states["resilient"] / 100.0
            self._boost_functions(new_modulation, ["perseverance", "problem_solving", "adaptation"], 0.5 * intensity)
        
        # Record modulation state
        modulation_record = {
            "timestamp": time.time(),
            "hormonal_levels": self.hormonal_params.get_current_levels(),
            "emotional_states": self.emotional_state.active_states.copy(),
            "modulation": new_modulation.copy()
        }
        
        self.modulation_history.append(modulation_record)
        
        # Limit history length
        if len(self.modulation_history) > 1000:
            self.modulation_history = self.modulation_history[-1000:]
            
        # Update current modulation
        self.current_modulation = new_modulation
        
        logger.debug(f"Updated functional modulation")
        return new_modulation
    
    def _boost_functions(self, 
                        modulation: Dict[str, float], 
                        functions: List[str], 
                        boost_amount: float) -> None:
        """Boost specific functions in the modulation dictionary.
        
        Args:
            modulation: Modulation dictionary to modify
            functions: List of functions to boost
            boost_amount: Amount to boost by (0-1)
        """
        for func in functions:
            if func in modulation:
                # Boost with diminishing returns
                current = modulation[func]
                # Max boost is 2.0 (double effectiveness)
                modulation[func] = min(2.0, current + (2.0 - current) * boost_amount)
    
    def _normalize_functions(self, 
                           modulation: Dict[str, float], 
                           strength: float) -> None:
        """Normalize functions by bringing extreme values closer to neutral.
        
        Args:
            modulation: Modulation dictionary to modify
            strength: Strength of normalization (0-1)
        """
        for func, value in modulation.items():
            # Move value closer to 1.0 based on strength
            modulation[func] = value + (1.0 - value) * strength
    
    def get_modulation_for_group(self, group_name: str) -> Dict[str, float]:
        """Get modulation values for a specific function group.
        
        Args:
            group_name: Name of the function group
            
        Returns:
            Dictionary of function modulations for the group
        """
        if group_name not in self.function_groups:
            return {}
            
        group_functions = self.function_groups[group_name]
        return {func: self.current_modulation.get(func, 1.0) for func in group_functions}
    
    def get_overall_modulation(self) -> Dict[str, float]:
        """Get average modulation for each function group.
        
        Returns:
            Dictionary of group modulations
        """
        result = {}
        
        for group, functions in self.function_groups.items():
            group_values = [self.current_modulation.get(func, 1.0) for func in functions]
            avg_value = sum(group_values) / len(group_values) if group_values else 1.0
            result[group] = avg_value
            
        return result
    
    def get_significant_modulations(self, threshold: float = 0.2) -> Dict[str, float]:
        """Get functions with significant modulation.
        
        Args:
            threshold: Threshold for significance (deviation from 1.0)
            
        Returns:
            Dictionary of significantly modulated functions
        """
        significant = {}
        
        for func, value in self.current_modulation.items():
            if abs(value - 1.0) >= threshold:
                significant[func] = value
                
        return significant
    
    def describe_current_modulation(self) -> str:
        """Get a textual description of the current functional modulation.
        
        Returns:
            Description string
        """
        significant = self.get_significant_modulations(0.3)
        
        if not significant:
            return "No significant functional modulation active."
            
        # Sort by deviation from neutral
        sorted_mods = sorted(significant.items(), key=lambda x: abs(x[1] - 1.0), reverse=True)
        
        lines = ["Current functional modulation:"]
        
        for func, value in sorted_mods[:5]:  # Top 5 most significant
            if value > 1.0:
                direction = "enhanced"
                percentage = (value - 1.0) * 100
            else:
                direction = "reduced"
                percentage = (1.0 - value) * 100
                
            lines.append(f"- {func.replace('_', ' ').capitalize()}: {direction} by {percentage:.1f}%")
            
        # Add average group modulation
        group_mod = self.get_overall_modulation()
        
        lines.append("\nFunction group modulation:")
        for group, value in group_mod.items():
            if abs(value - 1.0) >= 0.1:  # Only show significant group modulation
                if value > 1.0:
                    direction = "enhanced"
                    percentage = (value - 1.0) * 100
                else:
                    direction = "reduced"
                    percentage = (1.0 - value) * 100
                    
                lines.append(f"- {group.capitalize()}: {direction} by {percentage:.1f}%")
            
        return "\n".join(lines)


class HomeostaticRegulator:
    """Prevents extreme or negative hormonal states through homeostatic regulation."""
    
    def __init__(self, 
                 hormonal_params: HormonalParameters,
                 emotional_state: EmotionalState,
                 check_interval: float = 60.0):
        """Initialize the homeostatic regulator.
        
        Args:
            hormonal_params: Hormonal parameters
            emotional_state: Emotional state
            check_interval: Interval between regulation checks in seconds
        """
        self.hormonal_params = hormonal_params
        self.emotional_state = emotional_state
        self.check_interval = check_interval
        self.running = False
        self.regulation_task = None
        self.regulation_history = []
        
        # Define homeostatic rules
        self.homeostatic_rules = [
            {
                "name": "prevent_cortisol_spike",
                "description": "Prevent extreme cortisol levels",
                "condition": lambda p: p.get("cortisol", 0) > 80,
                "action": lambda: {"cortisol": -15.0, "endorphin": 10.0},
                "priority": 10
            },
            {
                "name": "prevent_adrenaline_spike",
                "description": "Prevent extreme adrenaline levels",
                "condition": lambda p: p.get("adrenaline", 0) > 85,
                "action": lambda: {"adrenaline": -15.0, "serotonin": 10.0},
                "priority": 10
            },
            {
                "name": "low_dopamine_correction",
                "description": "Correct low dopamine levels",
                "condition": lambda p: p.get("dopamine", 0) < 20,
                "action": lambda: {"dopamine": 10.0},
                "priority": 8
            },
            {
                "name": "low_serotonin_correction",
                "description": "Correct low serotonin levels",
                "condition": lambda p: p.get("serotonin", 0) < 25,
                "action": lambda: {"serotonin": 10.0},
                "priority": 8
            },
            {
                "name": "balance_restoration",
                "description": "Restore balance when multiple parameters are extreme",
                "condition": lambda p: sum(1 for v in p.values() if v < 20 or v > 80) >= 3,
                "action": lambda: {k: (50 - v) * 0.2 for k, v in self.hormonal_params.get_current_levels().items()},
                "priority": 9
            }
        ]
        
        logger.info("Homeostatic regulator initialized")
    
    async def start(self) -> None:
        """Start the homeostatic regulation process."""
        if self.running:
            return
            
        self.running = True
        self.regulation_task = asyncio.create_task(self._regulation_loop())
        
        logger.info("Homeostatic regulation started")
    
    async def stop(self) -> None:
        """Stop the homeostatic regulation process."""
        if not self.running:
            return
            
        self.running = False
        
        if self.regulation_task:
            self.regulation_task.cancel()
            try:
                await self.regulation_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Homeostatic regulation stopped")
    
    async def _regulation_loop(self) -> None:
        """Background loop for homeostatic regulation."""
        while self.running:
            try:
                # Check for regulation needs
                await self.check_and_regulate()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Error in homeostatic regulation: {e}")
                await asyncio.sleep(5.0)  # Longer pause on error
    
    async def check_and_regulate(self) -> bool:
        """Check for regulation needs and apply if necessary.
        
        Returns:
            Whether regulation was applied
        """
        # Get current parameters
        current_levels = self.hormonal_params.get_current_levels()
        
        # Check rules
        triggered_rules = []
        
        for rule in self.homeostatic_rules:
            if rule["condition"](current_levels):
                triggered_rules.append(rule)
                
        if not triggered_rules:
            return False  # No regulation needed
            
        # Sort by priority
        triggered_rules.sort(key=lambda r: r["priority"], reverse=True)
        
        # Apply the highest priority rule
        rule = triggered_rules[0]
        changes = rule["action"]()
        
        # Record regulation
        regulation_record = {
            "timestamp": time.time(),
            "rule_applied": rule["name"],
            "description": rule["description"],
            "before_levels": current_levels.copy(),
            "changes": changes.copy()
        }
        
        # Apply changes
        results = self.hormonal_params.update_parameters(changes, persist_ratio=0.05)
        
        # Update record with results
        regulation_record["after_levels"] = self.hormonal_params.get_current_levels()
        regulation_record["results"] = {k: (old, new) for k, (old, new) in results.items()}
        
        self.regulation_history.append(regulation_record)
        
        # Limit history length
        if len(self.regulation_history) > 1000:
            self.regulation_history = self.regulation_history[-1000:]
            
        # Update emotional state
        self.emotional_state.update_state()
        
        logger.info(f"Applied homeostatic regulation: {rule['name']}")
        return True
    
    def get_regulation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent regulation history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of regulation records
        """
        return list(self.regulation_history)[-limit:]
    
    def describe_recent_regulation(self) -> str:
        """Get a description of recent regulation activity.
        
        Returns:
            Description string
        """
        recent = self.get_regulation_history(5)
        
        if not recent:
            return "No recent homeostatic regulation activity."
            
        lines = ["Recent homeostatic regulation:"]
        
        for record in recent:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record["timestamp"]))
            lines.append(f"- {timestamp}: {record['description']}")
            
            # Add parameter changes
            for param, (old, new) in record.get("results", {}).items():
                change = new - old
                direction = "increased" if change > 0 else "decreased"
                lines.append(f"  * {param}: {old:.1f} → {new:.1f} ({direction} by {abs(change):.1f})")
                
        return "\n".join(lines)


class SelfRegulation:
    """Allows the system to consciously influence its emotional state."""
    
    def __init__(self, 
                 hormonal_params: HormonalParameters,
                 emotional_state: EmotionalState):
        """Initialize the self-regulation system.
        
        Args:
            hormonal_params: Hormonal parameters
            emotional_state: Emotional state
        """
        self.hormonal_params = hormonal_params
        self.emotional_state = emotional_state
        
        # Define target states
        self.target_states = {
            "creative": {
                "description": "Enhanced creativity and exploration",
                "parameter_changes": {
                    "dopamine": 20.0,
                    "cortisol": -10.0,
                    "endorphin": 10.0
                }
            },
            "analytical": {
                "description": "Enhanced analytical and logical capabilities",
                "parameter_changes": {
                    "serotonin": 15.0,
                    "cortisol": 5.0,
                    "dopamine": -5.0
                }
            },
            "empathetic": {
                "description": "Enhanced empathy and emotional understanding",
                "parameter_changes": {
                    "oxytocin": 20.0,
                    "serotonin": 10.0,
                    "adrenaline": -10.0
                }
            },
            "focused": {
                "description": "Enhanced focus and concentration",
                "parameter_changes": {
                    "dopamine": 10.0,
                    "serotonin": 15.0,
                    "adrenaline": 5.0
                }
            },
            "calm": {
                "description": "Reduced stress and increased stability",
                "parameter_changes": {
                    "serotonin": 15.0,
                    "cortisol": -15.0,
                    "adrenaline": -10.0,
                    "endorphin": 10.0
                }
            },
            "balanced": {
                "description": "Balanced state with no extreme parameters",
                "parameter_changes": {
                    # Will be calculated dynamically
                }
            }
        }
        
        # Regulation history
        self.regulation_history = []
        
        logger.info("Self-regulation system initialized")
    
    async def regulate_towards_state(self, 
                                   target_state: str,
                                   intensity: float = 1.0) -> Dict[str, Any]:
        """Regulate hormonal parameters towards a target state.
        
        Args:
            target_state: Name of the target state
            intensity: Intensity of regulation (0-1)
            
        Returns:
            Regulation results
        """
        if target_state not in self.target_states:
            logger.warning(f"Unknown target state: {target_state}")
            return {"success": False, "error": "Unknown target state"}
            
        # Get target state definition
        state_def = self.target_states[target_state]
        
        # For balanced state, calculate changes dynamically
        if target_state == "balanced":
            parameter_changes = self._calculate_balance_changes()
        else:
            # Scale changes by intensity
            parameter_changes = {
                param: change * intensity 
                for param, change in state_def["parameter_changes"].items()
            }
        
        # Record regulation attempt
        regulation_record = {
            "timestamp": time.time(),
            "target_state": target_state,
            "intensity": intensity,
            "before_levels": self.hormonal_params.get_current_levels(),
            "parameter_changes": parameter_changes.copy()
        }
        
        # Apply changes
        results = self.hormonal_params.update_parameters(parameter_changes, persist_ratio=0.1)
        
        # Update record with results
        regulation_record["after_levels"] = self.hormonal_params.get_current_levels()
        regulation_record["results"] = {k: (old, new) for k, (old, new) in results.items()}
        
        self.regulation_history.append(regulation_record)
        
        # Limit history length
        if len(self.regulation_history) > 1000:
            self.regulation_history = self.regulation_history[-1000:]
            
        # Update emotional state
        self.emotional_state.update_state()
        
        logger.info(f"Applied self-regulation towards {target_state} state with intensity {intensity}")
        
        return {
            "success": True,
            "target_state": target_state,
            "intensity": intensity,
            "changes": results,
            "current_state": self.emotional_state.get_state_description()
        }
    
    def _calculate_balance_changes(self) -> Dict[str, float]:
        """Calculate parameter changes needed to achieve balance.
        
        Returns:
            Dictionary of parameter changes
        """
        current_levels = self.hormonal_params.get_current_levels()
        changes = {}
        
        for param, value in current_levels.items():
            # Target is middle of the range (50)
            target = 50.0
            
            # Calculate change needed (stronger for more extreme values)
            distance = target - value
            
            # Apply more correction for more extreme values
            if abs(distance) > 30:
                change_factor = 0.3  # Stronger correction
            elif abs(distance) > 15:
                change_factor = 0.2  # Medium correction
            else:
                change_factor = 0.1  # Mild correction
                
            change = distance * change_factor
            
            if abs(change) >= 1.0:  # Only apply significant changes
                changes[param] = change
                
        return changes
    
    def is_state_achievable(self, 
                          target_state: str,
                          threshold: float = 0.7) -> Dict[str, Any]:
        """Check if a target state is achievable from the current state.
        
        Args:
            target_state: Name of the target state
            threshold: Threshold for considering state achievable (0-1)
            
        Returns:
            Dictionary with achievability assessment
        """
        if target_state not in self.target_states:
            return {"achievable": False, "reason": "Unknown target state"}
            
        # For balanced state, always achievable
        if target_state == "balanced":
            return {"achievable": True, "reason": "Balance can always be improved"}
            
        # Get current active emotional states
        active_states = self.emotional_state.active_states
        
        # If already in this state, it's achievable
        if target_state in active_states and active_states[target_state] >= 50:
            return {
                "achievable": True, 
                "reason": f"Already in {target_state} state",
                "current_intensity": active_states[target_state]
            }
            
        # Check parameter compatibility
        current_levels = self.hormonal_params.get_current_levels()
        state_def = self.target_states[target_state]
        parameter_changes = state_def.get("parameter_changes", {})
        
        # Check if any parameters would exceed limits
        for param, change in parameter_changes.items():
            if param in current_levels:
                new_value = current_levels[param] + change
                
                param_def = self.hormonal_params.parameter_definitions.get(param, {})
                min_val = param_def.get("min_value", 0)
                max_val = param_def.get("max_value", 100)
                
                if new_value < min_val or new_value > max_val:
                    return {
                        "achievable": False,
                        "reason": f"Parameter {param} would exceed limits",
                        "current_value": current_levels[param],
                        "change": change,
                        "new_value": new_value,
                        "limit_exceeded": "min" if new_value < min_val else "max"
                    }
        
        # Calculate approximate achievability
        # In a real implementation, this would use more sophisticated prediction
        similar_states = [s for s in self.emotional_state.emotional_states
                        if target_state in s or s in target_state]
        
        if similar_states:
            # Higher achievability if similar states are active
            similar_active = [s for s in similar_states if s in active_states]
            if similar_active:
                return {
                    "achievable": True,
                    "reason": f"Similar states already active: {', '.join(similar_active)}",
                    "similar_states": similar_active
                }
                
        # Default to achievable with standard adjustment
        return {
            "achievable": True,
            "reason": "State appears achievable through parameter adjustment",
            "required_changes": list(parameter_changes.keys())
        }
    
    def get_ideal_state_for_task(self, task_type: str) -> Optional[str]:
        """Get the ideal emotional state for a specific task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Name of ideal target state or None if unknown task
        """
        # Map task types to ideal states
        task_state_map = {
            "creative_writing": "creative",
            "brainstorming": "creative",
            "problem_solving": "creative",
            "data_analysis": "analytical",
            "logical_reasoning": "analytical",
            "pattern_recognition": "analytical",
            "emotional_support": "empathetic",
            "counseling": "empathetic",
            "social_interaction": "empathetic",
            "detailed_work": "focused",
            "coding": "focused",
            "research": "focused",
            "meditation": "calm",
            "relaxation": "calm",
            "general": "balanced"
        }
        
        # Fuzzy matching for task type
        for known_task, state in task_state_map.items():
            if known_task in task_type.lower() or task_type.lower() in known_task:
                return state
                
        # Default to balanced for unknown tasks
        return "balanced"
    
    def get_current_regulation_ability(self) -> Dict[str, float]:
        """Assess current ability to self-regulate in different directions.
        
        Returns:
            Dictionary of regulation abilities (0-1)
        """
        # Get current state
        current_levels = self.hormonal_params.get_current_levels()
        
        abilities = {}
        
        for state_name, state_def in self.target_states.items():
            if state_name == "balanced":
                # Ability to balance is higher when more parameters are extreme
                extremes = sum(1 for v in current_levels.values() if v < 20 or v > 80)
                abilities[state_name] = min(1.0, extremes / len(current_levels) * 2)
                continue
                
            # Calculate average headroom for required changes
            parameter_changes = state_def.get("parameter_changes", {})
            headrooms = []
            
            for param, change in parameter_changes.items():
                if param in current_levels:
                    current = current_levels[param]
                    
                    param_def = self.hormonal_params.parameter_definitions.get(param, {})
                    min_val = param_def.get("min_value", 0)
                    max_val = param_def.get("max_value", 100)
                    
                    # Calculate headroom in the direction of change
                    if change > 0:
                        headroom = (max_val - current) / change
                    elif change < 0:
                        headroom = (current - min_val) / abs(change)
                    else:
                        headroom = 1.0
                        
                    headrooms.append(min(1.0, headroom))
            
            # Overall ability is the minimum headroom (bottleneck)
            if headrooms:
                abilities[state_name] = min(headrooms)
            else:
                abilities[state_name] = 0.0
                
        return abilities
    
    def get_regulation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent regulation history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of regulation records
        """
        return list(self.regulation_history)[-limit:]
    
    def describe_regulation_options(self) -> str:
        """Get a description of available regulation options.
        
        Returns:
            Description string
        """
        # Get current ability to regulate
        abilities = self.get_current_regulation_ability()
        
        lines = ["Current self-regulation options:"]
        
        for state, ability in abilities.items():
            state_def = self.target_states.get(state, {})
            description = state_def.get("description", "Unknown state")
            
            if ability < 0.3:
                availability = "Limited capacity"
            elif ability < 0.7:
                availability = "Moderate capacity"
            else:
                availability = "Full capacity"
                
            lines.append(f"- {state.capitalize()}: {description}")
            lines.append(f"  * {availability} ({ability:.1%} potential)")
            
        # Add current emotional state
        lines.append("\nCurrent emotional state:")
        lines.append(self.emotional_state.get_state_description())
            
        return "\n".join(lines)


class HormonalSystem:
    """Main class that integrates all components of the hormonal system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hormonal system.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.hormonal_params = HormonalParameters(self.config)
        self.emotional_state = EmotionalState(self.hormonal_params)
        self.functional_modulator = FunctionalModulator(self.hormonal_params, self.emotional_state)
        self.homeostatic_regulator = HomeostaticRegulator(
            self.hormonal_params,
            self.emotional_state,
            check_interval=self.config.get("homeostatic_check_interval", 60.0)
        )
        self.self_regulation = SelfRegulation(self.hormonal_params, self.emotional_state)
        
        # Initialize emotional state
        self.emotional_state.update_state()
        
        # Update modulation
        self.functional_modulator.update_modulation()
        
        logger.info("Hormonal system initialized")
    
    async def start(self) -> None:
        """Start the hormonal system background processes."""
        await self.homeostatic_regulator.start()
        logger.info("Hormonal system started")
    
    async def stop(self) -> None:
        """Stop the hormonal system background processes."""
        await self.homeostatic_regulator.stop()
        logger.info("Hormonal system stopped")
    
    async def process_stimulus(self, 
                             stimulus_type: str, 
                             stimulus_content: Any,
                             intensity: float = 1.0) -> Dict[str, Any]:
        """Process a stimulus and update hormonal parameters.
        
        Args:
            stimulus_type: Type of stimulus (e.g., "query", "feedback", etc.)
            stimulus_content: Content of the stimulus
            intensity: Intensity factor for the stimulus (0-1)
            
        Returns:
            Processing results
        """
        # Calculate hormonal response based on stimulus type and content
        hormonal_response = await self._calculate_hormonal_response(
            stimulus_type, 
            stimulus_content,
            intensity
        )
        
        # Record before state
        before_state = {
            "hormonal_levels": self.hormonal_params.get_current_levels(),
            "emotional_states": self.emotional_state.active_states.copy()
        }
        
        # Apply hormonal changes
        parameter_changes = hormonal_response["parameter_changes"]
        results = self.hormonal_params.update_parameters(
            parameter_changes,
            persist_ratio=hormonal_response.get("persist_ratio", 0.1)
        )
        
        # Update emotional state
        self.emotional_state.update_state()
        
        # Record emotional response
        self.emotional_state.record_emotional_response(
            stimulus_type,
            stimulus_content,
            {
                "hormonal_response": hormonal_response,
                "parameter_changes": parameter_changes,
                "results": results
            }
        )
        
        # Update functional modulation
        modulation = self.functional_modulator.update_modulation()
        
        # Check for homeostatic regulation
        regulation_applied = await self.homeostatic_regulator.check_and_regulate()
        
        # Prepare result
        result = {
            "stimulus_type": stimulus_type,
            "before_state": before_state,
            "parameter_changes": {k: (old, new) for k, (old, new) in results.items()},
            "current_emotional_state": self.emotional_state.active_states.copy(),
            "dominant_state": self.emotional_state.get_dominant_state(),
            "modulation_applied": any(abs(v - 1.0) > 0.1 for v in modulation.values()),
            "significant_modulations": self.functional_modulator.get_significant_modulations(),
            "homeostatic_regulation_applied": regulation_applied
        }
        
        logger.info(f"Processed {stimulus_type} stimulus: {self.emotional_state.get_state_description()}")
        return result
    
    async def _calculate_hormonal_response(self,
                                         stimulus_type: str,
                                         stimulus_content: Any,
                                         intensity: float) -> Dict[str, Any]:
        """Calculate hormonal response to a stimulus.
        
        Args:
            stimulus_type: Type of stimulus
            stimulus_content: Content of the stimulus
            intensity: Intensity factor
            
        Returns:
            Hormonal response data
        """
        # Default empty response
        response = {
            "parameter_changes": {},
            "persist_ratio": 0.1
        }
        
        # Recall similar responses from emotional memory
        similar_responses = self.emotional_state.recall_similar_responses(
            stimulus_type,
            stimulus_content
        )
        
        # If we have similar responses, use them to inform our response
        if similar_responses:
            # In a real implementation, this would use more sophisticated analysis
            # For now, use a simple averaging approach
            
            # Get average parameter changes from similar responses
            all_changes = {}
            
            for memory in similar_responses:
                changes = memory.get("response_data", {}).get("parameter_changes", {})
                
                for param, change in changes.items():
                    if param not in all_changes:
                        all_changes[param] = []
                        
                    all_changes[param].append(change)
            
            # Calculate average changes
            avg_changes = {}
            
            for param, changes in all_changes.items():
                if changes:
                    avg_changes[param] = sum(changes) / len(changes) * intensity
                    
            response["parameter_changes"] = avg_changes
            response["based_on_memory"] = True
            
            return response
        
        # If no similar responses, calculate based on stimulus type
        if stimulus_type == "query":
            # Analyze content for keywords
            content_str = str(stimulus_content).lower()
            
            # Simple keyword-based approach
            changes = {}
            
            # Creative topics increase dopamine
            creative_keywords = ["creative", "imagine", "story", "design", "art", "novel", "innovative"]
            if any(keyword in content_str for keyword in creative_keywords):
                changes["dopamine"] = 10.0 * intensity
                
            # Technical/analytical topics increase cortisol (focus) and serotonin
            analytical_keywords = ["analyze", "calculate", "technical", "problem", "solution", "data", "code"]
            if any(keyword in content_str for keyword in analytical_keywords):
                changes["cortisol"] = 7.0 * intensity
                changes["serotonin"] = 8.0 * intensity
                
            # Emotional topics increase oxytocin
            emotional_keywords = ["feel", "emotion", "relationship", "love", "friend", "family", "support"]
            if any(keyword in content_str for keyword in emotional_keywords):
                changes["oxytocin"] = 12.0 * intensity
                
            # Urgent queries increase adrenaline
            urgent_keywords = ["urgent", "immediately", "quickly", "emergency", "critical", "important"]
            if any(keyword in content_str for keyword in urgent_keywords):
                changes["adrenaline"] = 15.0 * intensity
                
            # Default mild response for any query
            if not changes:
                changes["dopamine"] = 3.0 * intensity
                changes["adrenaline"] = 5.0 * intensity
                
            response["parameter_changes"] = changes
            
        elif stimulus_type == "feedback":
            # Analyze feedback sentiment
            content_str = str(stimulus_content).lower()
            
            # Simple sentiment analysis
            positive_keywords = ["good", "great", "excellent", "thank", "helpful", "amazing", "perfect"]
            negative_keywords = ["bad", "wrong", "incorrect", "not helpful", "useless", "confused", "error"]
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in content_str)
            negative_count = sum(1 for keyword in negative_keywords if keyword in content_str)
            
            sentiment = (positive_count - negative_count) * intensity
            
            changes = {}
            
            if sentiment > 0:
                # Positive feedback increases dopamine and serotonin
                changes["dopamine"] = 8.0 * sentiment
                changes["serotonin"] = 6.0 * sentiment
                changes["endorphin"] = 5.0 * sentiment
            elif sentiment < 0:
                # Negative feedback increases cortisol and decreases dopamine
                changes["cortisol"] = 5.0 * abs(sentiment)
                changes["dopamine"] = -3.0 * abs(sentiment)
                # But also increases endorphin for resilience
                changes["endorphin"] = 4.0 * abs(sentiment)
            
            response["parameter_changes"] = changes
            
        else:
            # Default minimal response for unknown stimulus types
            response["parameter_changes"] = {
                "dopamine": 2.0 * intensity,
                "cortisol": 1.0 * intensity
            }
            
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the complete status of the hormonal system.
        
        Returns:
            Status dictionary
        """
        return {
            "timestamp": time.time(),
            "hormonal_levels": self.hormonal_params.get_current_levels(),
            "base_levels": self.hormonal_params.get_base_levels(),
            "emotional_state": {
                "active_states": self.emotional_state.active_states.copy(),
                "dominant_state": self.emotional_state.get_dominant_state(),
                "description": self.emotional_state.get_state_description()
            },
            "functional_modulation": {
                "significant": self.functional_modulator.get_significant_modulations(),
                "group_modulation": self.functional_modulator.get_overall_modulation(),
                "description": self.functional_modulator.describe_current_modulation()
            },
            "regulation": {
                "recent_homeostatic": self.homeostatic_regulator.get_regulation_history(3),
                "recent_self_regulation": self.self_regulation.get_regulation_history(3),
                "regulation_options": self.self_regulation.get_current_regulation_ability()
            }
        }
    
    def get_state_description(self) -> str:
        """Get a human-readable description of the current system state.
        
        Returns:
            Description string
        """
        lines = ["=== Hormonal System Status ==="]
        
        # Add emotional state description
        lines.append("\n== Emotional State ==")
        lines.append(self.emotional_state.get_state_description())
        
        # Add key hormonal levels
        levels = self.hormonal_params.get_current_levels()
        lines.append("\n== Key Hormonal Parameters ==")
        for param, value in levels.items():
            lines.append(f"- {param.capitalize()}: {value:.1f}%")
            
        # Add functional modulation
        lines.append("\n== Functional Modulation ==")
        lines.append(self.functional_modulator.describe_current_modulation())
        
        # Add recent regulation
        lines.append("\n== Recent Regulation ==")
        recent_homeostatic = self.homeostatic_regulator.describe_recent_regulation()
        if "No recent" not in recent_homeostatic:
            lines.append(recent_homeostatic)
        else:
            lines.append("No recent homeostatic regulation.")
            
        return "\n".join(lines)
    
    def save_state(self, base_path: Path) -> bool:
        """Save the complete state of the hormonal system.
        
        Args:
            base_path: Base path for saving state files
            
        Returns:
            Success status
        """
        base_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save hormonal parameters
            params_success = self.hormonal_params.save_state(base_path / "hormonal_params.json")
            
            # Save emotional state
            emotional_success = self.emotional_state.save_state(base_path / "emotional_state.json")
            
            # Save system status
            status = self.get_system_status()
            with open(base_path / "system_status.json", 'w') as f:
                json.dump(status, f, indent=2)
                
            logger.info(f"Saved hormonal system state to {base_path}")
            return params_success and emotional_success
            
        except Exception as e:
            logger.error(f"Error saving hormonal system state: {e}")
            return False
    
    @classmethod
    def load_state(cls, base_path: Path, config: Optional[Dict[str, Any]] = None) -> Optional['HormonalSystem']:
        """Load hormonal system state from files.
        
        Args:
            base_path: Base path for loading state files
            config: Optional configuration to apply
            
        Returns:
            New HormonalSystem instance or None if loading failed
        """
        try:
            # Create new instance
            system = cls(config)
            
            # Load hormonal parameters
            params_path = base_path / "hormonal_params.json"
            if params_path.exists():
                params = HormonalParameters.load_state(params_path)
                if params:
                    system.hormonal_params = params
            
            # Load emotional state
            emotional_path = base_path / "emotional_state.json"
            if emotional_path.exists():
                emotional = EmotionalState.load_state(emotional_path, system.hormonal_params)
                if emotional:
                    system.emotional_state = emotional
            
            # Recreate dependent components with the loaded state
            system.functional_modulator = FunctionalModulator(system.hormonal_params, system.emotional_state)
            system.homeostatic_regulator = HomeostaticRegulator(
                system.hormonal_params,
                system.emotional_state,
                check_interval=system.config.get("homeostatic_check_interval", 60.0)
            )
            system.self_regulation = SelfRegulation(system.hormonal_params, system.emotional_state)
            
            # Update state
            system.emotional_state.update_state()
            system.functional_modulator.update_modulation()
            
            logger.info(f"Loaded hormonal system state from {base_path}")
            return system
            
        except Exception as e:
            logger.error(f"Error loading hormonal system state: {e}")
            return None
