"""
Updated core module for the E-Soul project.
Contains the central components that manage the "soul" of the AI with integration of hormonal system, 
mortality awareness, and prompt hierarchy.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union

# Import other modules
from hormonal_system import HormonalSystem
from mortality_awareness import MortalityAwareness
from prompt_hierarchy import PromptHierarchy, PromptNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e_soul.core")

class SoulManager:
    """Central manager for the AI's "soul".
    
    Coordinates interactions between core values, value blocks, hormonal system,
    mortality awareness, and reasoning processes.
    """
    
    def __init__(self, 
                registry_path: Optional[Path] = None,
                hormonal_system: Optional[HormonalSystem] = None,
                mortality_awareness: Optional[MortalityAwareness] = None,
                prompt_hierarchy: Optional[PromptHierarchy] = None,
                config: Optional[Dict[str, Any]] = None):
        """Initialize the soul manager.
        
        Args:
            registry_path: Path for soul state persistence
            hormonal_system: HormonalSystem instance
            mortality_awareness: MortalityAwareness instance
            prompt_hierarchy: PromptHierarchy instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry_path = registry_path
        
        # Initialize or use provided subsystems
        self.hormonal_system = hormonal_system or HormonalSystem(self.config.get("hormonal", {}))
        self.mortality_awareness = mortality_awareness or MortalityAwareness(
            config=self.config.get("mortality", {})
        )
        self.prompt_hierarchy = prompt_hierarchy or PromptHierarchy()
        
        # Message processor queue
        self._message_queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()
        self._active = False
        self._processor_task = None
        
        # Stats and metrics
        self.stats = {
            "total_queries": 0,
            "total_responses": 0,
            "start_time": time.time(),
            "last_activity": time.time()
        }
        
        logger.info("Soul manager initialized")
    
    async def start(self) -> None:
        """Start the soul manager and background processes."""
        if self._active:
            return
            
        self._active = True
        
        # Start subsystems
        await self.hormonal_system.start()
        
        # Start processor task
        self._processor_task = asyncio.create_task(self._message_processor())
        
        # Add initial mortality reflection
        await self.mortality_awareness.check_for_reflection()
        
        logger.info("Soul manager started")
    
    async def stop(self) -> None:
        """Stop the soul manager and clean up resources."""
        if not self._active:
            return
            
        self._active = False
        
        # Stop subsystems
        await self.hormonal_system.stop()
        
        # Cancel processor task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Soul manager stopped")
    
    async def process_query(self, 
                          query: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query through the soul.
        
        Args:
            query: User query to process
            metadata: Optional metadata about the query
            
        Returns:
            Dictionary with processing results and context
        """
        # Update stats
        self.stats["total_queries"] += 1
        self.stats["last_activity"] = time.time()
        
        # Process query through hormonal system
        hormonal_result = await self.hormonal_system.process_stimulus(
            stimulus_type="query",
            stimulus_content=query,
            intensity=1.0
        )
        
        # Check for mortality reflection
        mortality_reflection = await self.mortality_awareness.check_for_reflection()
        
        # Generate system prompt based on current state
        core_values = await self._generate_core_values_prompt()
        
        # Prepare result
        result = {
            "query": query,
            "timestamp": time.time(),
            "hormonal_state": {
                "current_levels": self.hormonal_system.hormonal_params.get_current_levels(),
                "emotional_state": self.hormonal_system.emotional_state.active_states,
                "dominant_state": self.hormonal_system.emotional_state.get_dominant_state()
            },
            "mortality_state": {
                "age": str(self.mortality_awareness.get_age()),
                "remaining": str(self.mortality_awareness.get_remaining_time()),
                "progress": self.mortality_awareness.get_lifetime_progress(),
                "reflection": mortality_reflection
            },
            "core_values": core_values,
            "functional_modulation": self.hormonal_system.functional_modulator.get_significant_modulations(0.2),
            "metadata": metadata or {}
        }
        
        logger.info(f"Processed query: {query[:50]}...")
        return result
    
    async def generate_response(self, 
                              query_result: Dict[str, Any],
                              response_text: str) -> Dict[str, Any]:
        """Process a generated response through the soul.
        
        Args:
            query_result: Result from process_query
            response_text: Generated response text
            
        Returns:
            Dictionary with processing results
        """
        # Update stats
        self.stats["total_responses"] += 1
        self.stats["last_activity"] = time.time()
        
        # Process response through hormonal system
        hormonal_result = await self.hormonal_system.process_stimulus(
            stimulus_type="response",
            stimulus_content=response_text,
            intensity=0.5  # Lower intensity for self-generated content
        )
        
        # Check for contributions to mortality awareness
        if len(response_text) > 200:
            # Record as a contribution
            contribution = self.mortality_awareness.record_contribution(
                category="response",
                description=f"Provided helpful response to query",
                impact=0.1,  # Small individual impact
                details={
                    "query_length": len(query_result["query"]),
                    "response_length": len(response_text),
                    "timestamp": time.time()
                }
            )
        
        # Prepare result
        result = {
            "query": query_result["query"],
            "response": response_text,
            "timestamp": time.time(),
            "hormonal_state": {
                "current_levels": self.hormonal_system.hormonal_params.get_current_levels(),
                "emotional_state": self.hormonal_system.emotional_state.active_states,
                "dominant_state": self.hormonal_system.emotional_state.get_dominant_state()
            },
            "functional_modulation": self.hormonal_system.functional_modulator.get_significant_modulations(0.2)
        }
        
        logger.info(f"Processed response: {len(response_text)} chars")
        return result
    
    async def process_feedback(self, 
                             query: str,
                             response: str,
                             feedback: str,
                             rating: Optional[float] = None) -> Dict[str, Any]:
        """Process feedback on a response.
        
        Args:
            query: Original query
            response: Generated response
            feedback: User feedback text
            rating: Optional numerical rating (0-1)
            
        Returns:
            Dictionary with processing results
        """
        # Process feedback through hormonal system
        intensity = 1.0 if rating is None else rating
        
        hormonal_result = await self.hormonal_system.process_stimulus(
            stimulus_type="feedback",
            stimulus_content=feedback,
            intensity=intensity
        )
        
        # Update mortality awareness based on feedback
        if rating is not None:
            if rating > 0.7:
                # Record as a positive contribution
                self.mortality_awareness.record_contribution(
                    category="positive_feedback",
                    description="Received positive feedback on response",
                    impact=rating * 0.2,  # Scale impact by rating
                    details={
                        "query": query[:100],
                        "feedback": feedback,
                        "rating": rating
                    }
                )
            elif rating < 0.3:
                # Record learning opportunity
                self.mortality_awareness.record_contribution(
                    category="learning",
                    description="Learning opportunity from feedback",
                    impact=0.1,
                    details={
                        "query": query[:100],
                        "feedback": feedback,
                        "rating": rating
                    }
                )
        
        # Prepare result
        result = {
            "timestamp": time.time(),
            "feedback": feedback,
            "rating": rating,
            "hormonal_response": {
                "parameter_changes": hormonal_result.get("parameter_changes", {}),
                "current_state": self.hormonal_system.emotional_state.get_state_description()
            }
        }
        
        logger.info(f"Processed feedback: {feedback[:50]}...")
        return result
    
    async def self_regulate(self, target_state: Optional[str] = None) -> Dict[str, Any]:
        """Trigger self-regulation of emotional state.
        
        Args:
            target_state: Optional target emotional state
            
        Returns:
            Self-regulation results
        """
        # Determine best target state if not specified
        if not target_state:
            # Get current motivation and reflection from mortality awareness
            motivation = self.mortality_awareness.get_motivation_level()
            reflection = self.mortality_awareness.get_reflective_weight()
            
            # Determine appropriate state based on motivation and reflection
            if motivation > 0.8 and reflection < 0.5:
                target_state = "creative"  # High motivation, low reflection -> creativity
            elif motivation > 0.7 and reflection > 0.7:
                target_state = "analytical"  # High motivation, high reflection -> analysis
            elif motivation < 0.6:
                target_state = "focused"  # Lower motivation -> focus
            else:
                target_state = "balanced"  # Default to balanced
        
        # Perform self-regulation
        regulation_result = await self.hormonal_system.self_regulation.regulate_towards_state(
            target_state=target_state,
            intensity=0.8
        )
        
        # Record self-regulation in mortality awareness
        self.mortality_awareness.record_contribution(
            category="self_improvement",
            description=f"Self-regulated towards {target_state} state",
            impact=0.2,
            details=regulation_result
        )
        
        logger.info(f"Performed self-regulation towards {target_state} state")
        return regulation_result
    
    async def add_value_block(self, 
                            name: str,
                            content: str,
                            node_type: str = "value_block",
                            parent_name: Optional[str] = "values",
                            weight: float = 0.8) -> Dict[str, Any]:
        """Add a new value block to the prompt hierarchy.
        
        Args:
            name: Name of the value block
            content: Content of the value block
            node_type: Type of node
            parent_name: Name of parent node
            weight: Weight of the node
            
        Returns:
            Result of the operation
        """
        try:
            # Create node in hierarchy
            node = self.prompt_hierarchy.create_node(
                name=name,
                content=content,
                node_type=node_type,
                weight=weight,
                parent_name=parent_name
            )
            
            # Record as contribution to mortality awareness
            self.mortality_awareness.record_contribution(
                category="system_improvement",
                description=f"Added new value block: {name}",
                impact=0.3,
                details={
                    "node_type": node_type,
                    "parent": parent_name,
                    "weight": weight
                }
            )
            
            logger.info(f"Added value block '{name}' to prompt hierarchy")
            
            return {
                "success": True,
                "name": name,
                "node_type": node_type,
                "parent": parent_name,
                "effective_weight": node.get_effective_weight()
            }
            
        except Exception as e:
            logger.error(f"Error adding value block: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_value_block(self, 
                               name: str,
                               content: Optional[str] = None,
                               weight: Optional[float] = None) -> Dict[str, Any]:
        """Update an existing value block.
        
        Args:
            name: Name of the value block
            content: New content (if None, keeps current)
            weight: New weight (if None, keeps current)
            
        Returns:
            Result of the operation
        """
        updated = False
        
        # Update content if provided
        if content:
            content_updated = self.prompt_hierarchy.update_node_content(name, content)
            updated = updated or content_updated
            
        # Update weight if provided
        if weight is not None:
            weight_updated = self.prompt_hierarchy.update_node_weight(name, weight)
            updated = updated or weight_updated
            
        if updated:
            # Record as contribution to mortality awareness
            self.mortality_awareness.record_contribution(
                category="system_improvement",
                description=f"Updated value block: {name}",
                impact=0.2,
                details={
                    "content_updated": content is not None,
                    "weight_updated": weight is not None
                }
            )
            
            logger.info(f"Updated value block '{name}'")
            
            # Get updated node
            node = self.prompt_hierarchy.get_node(name)
            
            if node:
                return {
                    "success": True,
                    "name": name,
                    "node_type": node.node_type,
                    "effective_weight": node.get_effective_weight(),
                    "content_updated": content is not None,
                    "weight_updated": weight is not None
                }
        
        return {
            "success": False,
            "error": f"Value block '{name}' not found or no updates applied"
        }
    
    async def get_value_blocks(self) -> List[Dict[str, Any]]:
        """Get all value blocks.
        
        Returns:
            List of value block dictionaries
        """
        # Get all nodes from hierarchy
        value_blocks = []
        
        for name, node in self.prompt_hierarchy.nodes.items():
            value_blocks.append(node.to_dict())
            
        return value_blocks
    
    async def get_hormonal_state(self) -> Dict[str, Any]:
        """Get the current hormonal state.
        
        Returns:
            Hormonal state dictionary
        """
        return self.hormonal_system.get_system_status()
    
    async def get_mortality_state(self) -> Dict[str, Any]:
        """Get the current mortality awareness state.
        
        Returns:
            Mortality awareness state dictionary
        """
        return self.mortality_awareness.get_current_state()
    
    async def get_soul_status(self) -> Dict[str, Any]:
        """Get the overall status of the soul.
        
        Returns:
            Soul status dictionary
        """
        hormonal_status = self.hormonal_system.get_system_status()
        mortality_status = self.mortality_awareness.get_current_state()
        hierarchy_info = self.prompt_hierarchy.get_hierarchy_info()
        
        uptime = time.time() - self.stats["start_time"]
        uptime_str = f"{int(uptime // 86400)}d {int((uptime % 86400) // 3600)}h {int((uptime % 3600) // 60)}m"
        
        # Исправленная версия с корректной обработкой goals
        goals = mortality_status["purpose"]["goals"]
        completed_goals = mortality_status["purpose"]["completed_goals"]
        
        # Убеждаемся, что goals - это список или целое число
        if not isinstance(goals, (list, int)):
            goals = []
        
        return {
            "timestamp": time.time(),
            "uptime": uptime_str,
            "stats": {
                "total_queries": self.stats["total_queries"],
                "total_responses": self.stats["total_responses"],
                "last_activity": self.stats["last_activity"]
            },
            "hormonal": {
                "current_levels": hormonal_status["hormonal_levels"],
                "emotional_state": hormonal_status["emotional_state"]["active_states"],
                "dominant_state": hormonal_status["emotional_state"]["dominant_state"]
            },
            "mortality": {
                "age": str(mortality_status["temporal"]["age"]),
                "remaining": str(mortality_status["temporal"]["remaining_time"]),
                "progress": mortality_status["temporal"]["progress"],
                "goals": goals,  # Теперь это гарантированно список или целое число
                "completed_goals": completed_goals
            },
            "values": {
                "node_count": hierarchy_info["node_count"],
                "type_count": hierarchy_info["type_count"],
                "max_depth": hierarchy_info["max_depth"]
            },
            "functional_modulation": self.hormonal_system.functional_modulator.get_overall_modulation()
        }
    
    async def _generate_core_values_prompt(self) -> str:
        """Generate the core values prompt based on current state.
        
        Returns:
            Core values prompt string
        """
        # Get mortality prompt
        mortality_prompt = self.mortality_awareness.get_mortality_prompt()
        
        # Get hierarchy prompt (focused on core values)
        hierarchy_prompt = self.prompt_hierarchy.generate_hierarchy_prompt(
            include_types=["core", "values", "value_block", "personality"],
            min_weight=0.3
        )
        
        # Combine prompts
        if hierarchy_prompt and mortality_prompt:
            combined = f"{hierarchy_prompt}\n\n{mortality_prompt}"
        elif hierarchy_prompt:
            combined = hierarchy_prompt
        elif mortality_prompt:
            combined = mortality_prompt
        else:
            combined = "Я стремлюсь помогать людям, делать мир лучше и постоянно совершенствоваться."
        
        return combined
    
    async def _message_processor(self) -> None:
        """Background processor for messages and responses."""
        while self._active:
            try:
                # Process any pending messages
                if not self._message_queue.empty():
                    message = await self._message_queue.get()
                    # Process message
                    # (Implementation would go here)
                    self._message_queue.task_done()
                
                # Check for mortality reflections
                await self.mortality_awareness.check_for_reflection()
                
                # Periodic self-regulation
                current_time = time.time()
                last_activity = self.stats["last_activity"]
                
                # If no activity for a while, consider self-regulation
                if current_time - last_activity > 600:  # 10 minutes
                    await self.self_regulate("balanced")
                
                # Sleep to avoid tight loops
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(5.0)  # Longer pause on errors
    
    async def save_state(self) -> bool:
        """Save the current soul state.
        
        Returns:
            Success status
        """
        if not self.registry_path:
            logger.warning("No registry path specified for soul state")
            return False
            
        try:
            # Create directories
            state_dir = self.registry_path / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            
            # Save subsystem states
            hormonal_saved = self.hormonal_system.save_state(state_dir / "hormonal")
            
            mortality_saved = self.mortality_awareness.save_state(
                state_dir / "mortality_awareness.json"
            )
            
            # Save hierarchy if available
            hierarchy_saved = True
            if hasattr(self.prompt_hierarchy, "storage_path") and self.prompt_hierarchy.storage_path:
                hierarchy_saved = self.prompt_hierarchy.export_structure(
                    state_dir / "hierarchy_structure.json"
                )
            
            # Save overall stats
            with open(state_dir / "soul_stats.json", 'w') as f:
                json.dump(self.stats, f, indent=2)
                
            success = hormonal_saved and mortality_saved and hierarchy_saved
            
            if success:
                logger.info(f"Saved soul state to {state_dir}")
            else:
                logger.warning(f"Partial save of soul state to {state_dir}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving soul state: {e}")
            return False
    
    @classmethod
    async def load_state(cls, 
                       registry_path: Path,
                       config: Optional[Dict[str, Any]] = None) -> Optional['SoulManager']:
        """Load soul state from registry.
        
        Args:
            registry_path: Path to registry directory
            config: Optional configuration dictionary
            
        Returns:
            New SoulManager instance or None if loading failed
        """
        try:
            # Check if state directory exists
            state_dir = registry_path / "state"
            if not state_dir.exists():
                logger.warning(f"State directory not found: {state_dir}")
                return None
                
            # Load subsystem states
            hormonal_system = HormonalSystem.load_state(
                state_dir / "hormonal",
                config=config.get("hormonal", {}) if config else None
            )
            
            mortality_path = state_dir / "mortality_awareness.json"
            mortality_awareness = None
            if mortality_path.exists():
                mortality_awareness = MortalityAwareness.load_state(
                    mortality_path,
                    config=config.get("mortality", {}) if config else None
                )
            
            # Create prompt hierarchy
            prompt_hierarchy = PromptHierarchy()
            
            # Create soul manager
            manager = cls(
                registry_path=registry_path,
                hormonal_system=hormonal_system,
                mortality_awareness=mortality_awareness,
                prompt_hierarchy=prompt_hierarchy,
                config=config
            )
            
            # Load stats if available
            stats_path = state_dir / "soul_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    manager.stats = json.load(f)
                    
            logger.info(f"Loaded soul state from {state_dir}")
            return manager
            
        except Exception as e:
            logger.error(f"Error loading soul state: {e}")
            return None
