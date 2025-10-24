"""Agent modules for Pokemon Emerald speedrunning agent."""

from collections import deque

from utils.vlm import VLM
from .action import action_step
from .memory import memory_step
from .perception import perception_step
from .planning import planning_step
from .simple import (
    SimpleAgent,
    configure_simple_agent_defaults,
    get_simple_agent,
    simple_mode_processing_multiprocess,
)


class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.
        
        Args:
            args: Command line arguments with agent configuration
        """
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"
        simple_mode = args.simple if args else False
        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent mode
        self.simple_mode = simple_mode
        if simple_mode:
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.simple_agent = get_simple_agent(self.vlm)
            print("   Mode: Simple (direct frame->action)")
        else:
            # Shared context across the four-module pipeline
            self.context = {
                "memory_context": "Initialized memory context",
                "current_plan": None,
                "recent_actions": deque(maxlen=25),
                "last_observation": None,
                "step_counter": 0,
            }
            print("   Mode: Four-module architecture")
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning'
        """
        if self.simple_mode:
            # Simple mode - delegate to SimpleAgent
            return self.simple_agent.step(game_state)
        else:
            # Four-module processing
            try:
                frame = game_state.get("frame") or game_state.get("visual", {}).get("screenshot")
                state_data = {k: v for k, v in game_state.items() if k != "frame"}

                # 1. Perception – understand what's happening in the frame/state
                observation_dict, slow_thinking_needed = perception_step(frame, state_data, self.vlm)
                observation_text = observation_dict.get("description", str(observation_dict))
                self.context["last_observation"] = observation_dict

                # 2. Memory – update rolling context with the latest observation/actions
                observation_entry = {
                    "frame_id": state_data.get("step_number", self.context["step_counter"]),
                    "observation": observation_text,
                    "state": state_data,
                }
                memory_context = memory_step(
                    self.context.get("memory_context", ""),
                    self.context.get("current_plan"),
                    list(self.context.get("recent_actions", [])),
                    [observation_entry],
                    self.vlm,
                )
                self.context["memory_context"] = memory_context

                # 3. Planning – refresh strategy if needed
                current_plan = planning_step(
                    memory_context,
                    self.context.get("current_plan"),
                    slow_thinking_needed,
                    state_data,
                    self.vlm,
                )
                self.context["current_plan"] = current_plan

                # 4. Action – pick button presses, track recent history
                recent_actions = list(self.context.get("recent_actions", []))
                actions = action_step(
                    memory_context,
                    current_plan,
                    observation_text,
                    frame,
                    state_data,
                    recent_actions,
                    self.vlm,
                )

                if actions is None:
                    actions = []
                elif isinstance(actions, str):
                    actions = [btn.strip() for btn in actions.split(",") if btn.strip()]
                else:
                    actions = list(actions)

                for action in actions:
                    self.context["recent_actions"].append(action)

                self.context["step_counter"] += 1

                return {
                    "action": actions,
                    "observation": observation_dict,
                    "plan": current_plan,
                    "memory_context": memory_context,
                }

            except Exception as e:
                print(f"❌ Agent error: {e}")
                return None


__all__ = [
    'Agent',
    'action_step',
    'memory_step', 
    'perception_step',
    'planning_step',
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults'
]
