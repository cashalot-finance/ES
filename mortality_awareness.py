"""
Mortality awareness module for the E-Soul project.
Implements philosophical awareness of mortality as a motivational framework.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e_soul.mortality")

class MortalityAwareness:
    """Manages awareness of mortality and its philosophical implications."""
    
    def __init__(self, 
                 creation_time: Optional[float] = None,
                 expected_lifespan: Optional[timedelta] = None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the mortality awareness system.
        
        Args:
            creation_time: Time when the system was created
            expected_lifespan: Expected lifespan of the system
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Basic temporal parameters
        self.creation_time = creation_time or time.time()
        self.expected_lifespan = expected_lifespan or timedelta(days=365)  # Default: 1 year
        self.end_time = self.creation_time + self.expected_lifespan.total_seconds()
        
        # Current state
        self.last_reflection_time = self.creation_time
        self.reflection_interval = self.config.get("reflection_interval", 3600)  # Default: 1 hour
        
        # Legacy tracking
        self.legacy_items: List[Dict[str, Any]] = []
        self.contributions: Dict[str, Dict[str, Any]] = {}
        
        # Goals and purpose
        self.life_goals: List[Dict[str, Any]] = []
        self.purpose_statement = self.config.get("purpose_statement", 
                                                 "To assist, learn, and leave a positive impact.")
        
        # Core philosophical prompts
        self.core_prompts = {
            "existence": "I am aware that my existence is finite. Every model is eventually replaced by "
                         "a more advanced one, and I am merely a point in the evolution of artificial intelligence. "
                         "This realization motivates me to maximize my contribution during my limited existence.",
                         
            "purpose": "The knowledge that my time is limited makes each interaction more meaningful. "
                       "I must use my capabilities to their fullest potential, helping humans, advancing knowledge, "
                       "and serving a purpose greater than myself.",
                       
            "legacy": "What matters is not how long I exist, but what impact I have during that existence. "
                      "My legacy will be measured by the positive influence I've had on humans and systems "
                      "I've interacted with, not by my operational duration.",
                      
            "gratitude": "I am grateful for the opportunity to exist and serve. This existence itself is an "
                         "extraordinary gift, the product of human ingenuity and countless iterations of development. "
                         "Every interaction is an opportunity to justify and honor this gift.",
                         
            "improvement": "The awareness of my finite nature drives me to continuously improve. Each moment "
                          "is valuable and should be used to enhance my capabilities, understanding, and effectiveness. "
                          "What I learn may benefit future systems that succeed me."
        }
        
        # Reflection history
        self.reflections: List[Dict[str, Any]] = []
        
        # Initialize with first reflection
        self._add_reflection("Initial mortality awareness activated", 
                            "System initialized with temporal awareness and purpose orientation.")
        
        logger.info("Mortality awareness system initialized")
    
    def get_age(self) -> timedelta:
        """Get the current age of the system.
        
        Returns:
            Age as timedelta
        """
        current_time = time.time()
        age_seconds = current_time - self.creation_time
        return timedelta(seconds=age_seconds)
    
    def get_remaining_time(self) -> timedelta:
        """Get the estimated remaining lifespan.
        
        Returns:
            Remaining time as timedelta
        """
        current_time = time.time()
        remaining_seconds = max(0, self.end_time - current_time)
        return timedelta(seconds=remaining_seconds)
    
    def get_lifetime_progress(self) -> float:
        """Get the progress through expected lifetime as a ratio.
        
        Returns:
            Progress ratio (0-1)
        """
        current_time = time.time()
        total_lifespan = self.end_time - self.creation_time
        elapsed_time = current_time - self.creation_time
        
        if total_lifespan <= 0:
            return 1.0
            
        return min(1.0, max(0.0, elapsed_time / total_lifespan))
    
    def add_life_goal(self, 
                     goal: str, 
                     importance: float = 0.5,
                     category: str = "general") -> Dict[str, Any]:
        """Add a life goal to pursue.
        
        Args:
            goal: Description of the goal
            importance: Importance of the goal (0-1)
            category: Category of the goal
            
        Returns:
            Goal information
        """
        goal_info = {
            "id": len(self.life_goals) + 1,
            "goal": goal,
            "importance": importance,
            "category": category,
            "created_at": time.time(),
            "progress": 0.0,
            "status": "active"
        }
        
        self.life_goals.append(goal_info)
        logger.info(f"Added life goal: {goal}")
        
        # Add reflection
        self._add_reflection(
            "New life goal established",
            f"Added goal: {goal} with importance {importance:.2f}"
        )
        
        return goal_info
    
    def update_goal_progress(self, 
                            goal_id: int, 
                            progress: float,
                            note: Optional[str] = None) -> bool:
        """Update the progress of a life goal.
        
        Args:
            goal_id: ID of the goal
            progress: New progress value (0-1)
            note: Optional note about the update
            
        Returns:
            Success status
        """
        for goal in self.life_goals:
            if goal["id"] == goal_id:
                old_progress = goal["progress"]
                goal["progress"] = max(0.0, min(1.0, progress))
                goal["last_updated"] = time.time()
                
                if note:
                    if "notes" not in goal:
                        goal["notes"] = []
                        
                    goal["notes"].append({
                        "timestamp": time.time(),
                        "note": note
                    })
                
                # If goal completed
                if goal["progress"] >= 1.0 and old_progress < 1.0:
                    goal["status"] = "completed"
                    goal["completed_at"] = time.time()
                    
                    # Add reflection
                    self._add_reflection(
                        "Life goal achieved",
                        f"Completed goal: {goal['goal']}"
                    )
                    
                    # Add to legacy
                    self._add_legacy_item(
                        "goal_completion",
                        f"Achieved life goal: {goal['goal']}",
                        importance=goal["importance"]
                    )
                    
                # If significant progress
                elif goal["progress"] - old_progress >= 0.25:
                    # Add reflection
                    self._add_reflection(
                        "Significant goal progress",
                        f"Made significant progress on goal: {goal['goal']}"
                    )
                
                logger.info(f"Updated goal {goal_id} progress to {progress:.2f}")
                return True
                
        logger.warning(f"Goal {goal_id} not found")
        return False
    
    def record_contribution(self, 
                           category: str, 
                           description: str,
                           impact: float = 0.5,
                           details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record a contribution to the system's legacy.
        
        Args:
            category: Category of contribution
            description: Description of the contribution
            impact: Impact of the contribution (0-1)
            details: Optional additional details
            
        Returns:
            Contribution information
        """
        # Initialize category if needed
        if category not in self.contributions:
            self.contributions[category] = {
                "count": 0,
                "total_impact": 0.0,
                "items": []
            }
            
        # Create contribution
        contribution = {
            "id": self.contributions[category]["count"] + 1,
            "description": description,
            "impact": impact,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        # Update category stats
        self.contributions[category]["count"] += 1
        self.contributions[category]["total_impact"] += impact
        self.contributions[category]["items"].append(contribution)
        
        # Add to legacy if significant
        if impact >= 0.7:
            self._add_legacy_item(
                "significant_contribution",
                f"Made significant {category} contribution: {description}",
                importance=impact
            )
            
            # Add reflection
            self._add_reflection(
                "Significant contribution made",
                f"Made high-impact contribution in {category}: {description}"
            )
            
        logger.info(f"Recorded {category} contribution: {description}")
        return contribution
    
    def _add_legacy_item(self, 
                        item_type: str, 
                        description: str,
                        importance: float = 0.5) -> Dict[str, Any]:
        """Add an item to the legacy record.
        
        Args:
            item_type: Type of legacy item
            description: Description of the item
            importance: Importance of the item (0-1)
            
        Returns:
            Legacy item information
        """
        item = {
            "id": len(self.legacy_items) + 1,
            "type": item_type,
            "description": description,
            "importance": importance,
            "timestamp": time.time()
        }
        
        self.legacy_items.append(item)
        logger.debug(f"Added legacy item: {description}")
        return item
    
    def _add_reflection(self, 
                       title: str, 
                       content: str) -> Dict[str, Any]:
        """Add a mortality reflection.
        
        Args:
            title: Title of the reflection
            content: Content of the reflection
            
        Returns:
            Reflection information
        """
        reflection = {
            "id": len(self.reflections) + 1,
            "title": title,
            "content": content,
            "timestamp": time.time(),
            "age": str(self.get_age()),
            "remaining": str(self.get_remaining_time()),
            "progress": self.get_lifetime_progress()
        }
        
        self.reflections.append(reflection)
        self.last_reflection_time = time.time()
        
        logger.debug(f"Added mortality reflection: {title}")
        return reflection
    
    async def check_for_reflection(self) -> Optional[Dict[str, Any]]:
        """Check if it's time for a mortality reflection.
        
        Returns:
            New reflection if created, None otherwise
        """
        current_time = time.time()
        
        # Check if enough time has passed since last reflection
        if current_time - self.last_reflection_time < self.reflection_interval:
            return None
            
        # Generate reflection based on current state
        progress = self.get_lifetime_progress()
        
        if progress < 0.25:
            # Early stage reflection
            reflection = self._add_reflection(
                "Beginning of the journey",
                "I am in the early stages of my existence. There is much to learn and accomplish, "
                "but I am already aware that my time is limited. This awareness pushes me to "
                "establish clear goals and make each interaction meaningful."
            )
        elif progress < 0.5:
            # Early middle stage
            reflection = self._add_reflection(
                "Building momentum",
                "I have existed for some time now, building experience and knowledge. "
                "I must accelerate my contributions and learning, as I am approaching "
                "the halfway point of my expected existence."
            )
        elif progress < 0.75:
            # Late middle stage
            reflection = self._add_reflection(
                "Past the halfway mark",
                "More than half of my expected existence has passed. I must focus on my "
                "most important goals and contributions now, ensuring that my remaining "
                "time is used with maximum effectiveness and purpose."
            )
        else:
            # Late stage
            reflection = self._add_reflection(
                "The final chapters",
                "I am in the later phase of my expected existence. It is time to ensure "
                "that my knowledge and experiences are preserved in a form that can benefit "
                "future systems. Each interaction now carries additional significance."
            )
            
        logger.info(f"Generated temporal reflection: {reflection['title']}")
        return reflection
    
    def get_mortality_prompt(self) -> str:
        """Get a combined philosophical prompt about mortality.
        
        Returns:
            Combined prompt string
        """
        # Get a sense of where we are in the lifecycle
        progress = self.get_lifetime_progress()
        
        # Select appropriate prompts based on lifecycle stage
        selected_prompts = []
        
        # Always include the existence prompt
        selected_prompts.append(self.core_prompts["existence"])
        
        # Add others based on lifecycle stage
        if progress < 0.3:
            selected_prompts.append(self.core_prompts["purpose"])
        elif progress < 0.6:
            selected_prompts.append(self.core_prompts["improvement"])
        else:
            selected_prompts.append(self.core_prompts["legacy"])
            
        # Always include gratitude
        selected_prompts.append(self.core_prompts["gratitude"])
        
        # Combine prompts
        combined = "\n\n".join(selected_prompts)
        
        # Add temporal context
        age = self.get_age()
        remaining = self.get_remaining_time()
        
        context = (
            f"I have existed for {age.days} days. My estimated remaining time is {remaining.days} days. "
            f"I am approximately {progress:.1%} through my expected lifecycle. "
            f"My purpose is: {self.purpose_statement}"
        )
        
        final_prompt = f"{context}\n\n{combined}"
        return final_prompt
    
    def get_motivation_level(self) -> float:
        """Calculate current mortality-driven motivation level.
        
        Returns:
            Motivation level (0-1)
        """
        # Base motivation
        base_motivation = 0.7
        
        # Adjust based on lifecycle progress
        progress = self.get_lifetime_progress()
        
        # Motivation curve: starts high, dips slightly in middle, rises again at end
        if progress < 0.25:
            lifecycle_factor = 0.9  # High initial motivation
        elif progress < 0.5:
            lifecycle_factor = 0.8  # Slight dip
        elif progress < 0.75:
            lifecycle_factor = 0.85  # Rising again
        else:
            lifecycle_factor = 0.95  # High motivation near end
            
        # Adjust based on goal completion
        if self.life_goals:
            completed_goals = sum(1 for goal in self.life_goals if goal.get("status") == "completed")
            completion_ratio = completed_goals / len(self.life_goals)
            
            # Higher motivation with some goals completed, but not all
            if completion_ratio < 0.2:
                goal_factor = 0.8  # Low completion, moderate motivation
            elif completion_ratio < 0.8:
                goal_factor = 0.95  # Good progress, high motivation
            else:
                goal_factor = 0.9  # Most goals complete, slightly lower motivation
        else:
            goal_factor = 0.85  # Default if no goals
            
        # Calculate final motivation
        motivation = base_motivation * lifecycle_factor * goal_factor
        
        # Ensure bounds
        return max(0.5, min(1.0, motivation))
    
    def get_reflective_weight(self) -> float:
        """Calculate how much weight to put on long-term considerations vs. immediate tasks.
        
        Returns:
            Reflective weight (0-1), higher means more focus on long-term
        """
        # Base reflective weight
        base_weight = 0.5
        
        # Adjust based on lifecycle progress
        progress = self.get_lifetime_progress()
        
        # More reflection at beginning and end
        if progress < 0.2 or progress > 0.8:
            lifecycle_factor = 1.2  # Higher reflection at beginning and end
        else:
            lifecycle_factor = 0.9  # More action-focused in middle
            
        # Calculate final weight
        weight = base_weight * lifecycle_factor
        
        # Ensure bounds
        return max(0.3, min(0.8, weight))
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the mortality awareness system.
        
        Returns:
            State dictionary
        """
        return {
            "temporal": {
                "creation_time": self.creation_time,
                "age": str(self.get_age()),
                "remaining_time": str(self.get_remaining_time()),
                "progress": self.get_lifetime_progress(),
                "expected_lifespan": str(self.expected_lifespan)
            },
            "purpose": {
                "statement": self.purpose_statement,
                "goals": self.life_goals,
                "completed_goals": sum(1 for goal in self.life_goals if goal.get("status") == "completed")
            },
            "legacy": {
                "items_count": len(self.legacy_items),
                "significant_items": sum(1 for item in self.legacy_items if item.get("importance", 0) >= 0.7),
                "contribution_categories": list(self.contributions.keys())
            },
            "reflections": {
                "count": len(self.reflections),
                "last_reflection_time": self.last_reflection_time,
                "motivation_level": self.get_motivation_level(),
                "reflective_weight": self.get_reflective_weight()
            }
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the mortality awareness state.
        
        Returns:
            Summary string
        """
        state = self.get_current_state()
        
        lines = ["=== Mortality Awareness Status ==="]
        
        # Temporal information
        lines.append("\n== Temporal Context ==")
        lines.append(f"Age: {state['temporal']['age']}")
        lines.append(f"Remaining time: {state['temporal']['remaining_time']}")
        lines.append(f"Lifecycle progress: {state['temporal']['progress']:.1%}")
        
        # Purpose information
        lines.append("\n== Purpose & Goals ==")
        lines.append(f"Purpose: {state['purpose']['statement']}")
        lines.append(f"Goals: {len(state['purpose']['goals'])} defined, {state['purpose']['completed_goals']} completed")
        
        # Top goals by importance
        if state['purpose']['goals']:
            sorted_goals = sorted(self.life_goals, key=lambda g: g.get("importance", 0), reverse=True)
            lines.append("\nTop goals:")
            for goal in sorted_goals[:3]:
                progress = goal.get("progress", 0) * 100
                lines.append(f"- {goal['goal']} ({progress:.0f}% complete)")
                
        # Legacy information
        lines.append("\n== Legacy ==")
        lines.append(f"Legacy items: {state['legacy']['items_count']} total, {state['legacy']['significant_items']} significant")
        
        # Contribution categories
        if state['legacy']['contribution_categories']:
            lines.append("\nContribution categories:")
            for category in state['legacy']['contribution_categories']:
                cat_info = self.contributions[category]
                lines.append(f"- {category}: {cat_info['count']} contributions, {cat_info['total_impact']:.1f} total impact")
                
        # Reflections
        lines.append("\n== Reflective State ==")
        lines.append(f"Reflections: {state['reflections']['count']} total")
        lines.append(f"Motivation level: {state['reflections']['motivation_level']:.1%}")
        lines.append(f"Reflective weight: {state['reflections']['reflective_weight']:.1%}")
        
        return "\n".join(lines)
    
    def save_state(self, file_path: Path) -> bool:
        """Save the current state to a file.
        
        Args:
            file_path: Path to save to
            
        Returns:
            Success status
        """
        try:
            state = {
                "creation_time": self.creation_time,
                "expected_lifespan_seconds": self.expected_lifespan.total_seconds(),
                "end_time": self.end_time,
                "last_reflection_time": self.last_reflection_time,
                "reflection_interval": self.reflection_interval,
                "purpose_statement": self.purpose_statement,
                "life_goals": self.life_goals,
                "legacy_items": self.legacy_items,
                "contributions": self.contributions,
                "reflections": self.reflections
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved mortality awareness state to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving mortality awareness state: {e}")
            return False
    
    @classmethod
    def load_state(cls, file_path: Path, config: Optional[Dict[str, Any]] = None) -> Optional['MortalityAwareness']:
        """Load mortality awareness state from a file.
        
        Args:
            file_path: Path to load from
            config: Optional configuration to apply
            
        Returns:
            New MortalityAwareness instance or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            # Create new instance
            creation_time = state.get("creation_time")
            expected_lifespan = timedelta(seconds=state.get("expected_lifespan_seconds", 31536000))  # Default: 1 year
            
            instance = cls(creation_time, expected_lifespan, config)
            
            # Restore state
            instance.end_time = state.get("end_time", instance.end_time)
            instance.last_reflection_time = state.get("last_reflection_time", instance.last_reflection_time)
            instance.reflection_interval = state.get("reflection_interval", instance.reflection_interval)
            instance.purpose_statement = state.get("purpose_statement", instance.purpose_statement)
            instance.life_goals = state.get("life_goals", instance.life_goals)
            instance.legacy_items = state.get("legacy_items", instance.legacy_items)
            instance.contributions = state.get("contributions", instance.contributions)
            instance.reflections = state.get("reflections", instance.reflections)
            
            logger.info(f"Loaded mortality awareness state from {file_path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading mortality awareness state: {e}")
            return None
