from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field
from langfuse import observe
from loguru import logger
import json

from llm.base import LLMClient
from tools.registry import ToolRegistry


class BaseAgentState(BaseModel):
    """
    Holds the evolving state of an agent's execution.
    """
    messages: list[dict] = Field(default_factory=list)
    is_finished: bool = False
    iteration: int = 0

    def add_message(self, role: str, content: str, **extra):
        msg = {"role": role, "content": content}
        # extra for example tool_id
        msg.update(extra)
        self.messages.append(msg)
        

class Agent(ABC):
    def __init__( self, llm: LLMClient, tool_registry: ToolRegistry, max_iterations: int = 100):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        # initial state to start with
        self.inital_state = BaseAgentState()

    @abstractmethod
    def start_point(self, *args, **kwargs) -> BaseAgentState:
        """ Start Point of State for example start of user query or anything """
        raise NotImplementedError()
    
    @abstractmethod
    def run(self, state: BaseAgentState) -> BaseAgentState:
        """ Run 1 Step/Iteration """
        raise NotImplementedError()

    def iterate(self, *args, **kwargs) -> BaseAgentState:
        state = self.start_point(*args, **kwargs)
        while not state.is_finished and state.iteration < self.max_iterations:
            logger.debug(f"Agent Iteration: {state.iteration}")
            state = self.run(state)
            state.iteration += 1

        return state

    # LLM WRAPPER
    @observe(name="llm-call", as_type="generation")
    def llm_generate(self, state: BaseAgentState):
        client_tools = self.tool_registry.to_client_tools(self.llm.config.provider)
        response_dict = self.llm.generate(state.messages, tools=client_tools)
        return response_dict
    
    # TOOL EXECUTION WRAPPER
    @observe(name="tool-call", as_type="tool")
    def call_tool(self, tool_call):
        try:
            func_name = tool_call['function']['name']
            args = json.loads(tool_call['function']['arguments'])
            
            tool_instance = self.tool_registry.get(func_name)
            if not tool_instance:
                raise ValueError(f"Tool {func_name} not found")
                
            result = tool_instance(**args)
            return {
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "content": str(result) # ensure content is string
            }
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "content": f"Error executing tool: {str(e)}"
            }
    