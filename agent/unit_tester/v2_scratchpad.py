from ..base import Agent, BaseAgentState, LLMClient, ToolRegistry
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from pathlib import Path
from loguru import logger
import json

class ScratchpadUnitTesterAgent(Agent):
    def __init__(self, llm: LLMClient,  max_iterations: int = 100):
        tool_registry = ToolRegistry()
        tool_registry.register_from_module(code_tools)
        tool_registry.register_from_module(file_tools)
        tool_registry.register_from_module(json_tools)
        
        super().__init__(llm, tool_registry, max_iterations)
        
        self.inital_state = BaseAgentState()
        
        prompt_path = Path("prompts/unit_tester_v2.txt")
        if prompt_path.exists():
            sys_template = prompt_path.read_text(encoding="utf-8")
        else:
            sys_template = "You are a QA Agent. Output JSON. Tools: {tools}"
            
        formatted_prompt = sys_template.replace("{tools}", self.tool_registry.to_string())
        
        self.inital_state.add_message(role="system", content=formatted_prompt)
    
    def start_point(self, user_query) -> BaseAgentState:
        # Use model_copy to ensure we don't mutate the base state template
        state = self.inital_state.model_copy(deep=True)
        state.add_message(role="user", content=user_query)
        return state
    
    def run(self, state: BaseAgentState) -> BaseAgentState:
        response = self.llm_generate(state)
        content = response.get("content") or ""
        tool_calls = response.get("tool_calls") or []
        
        persistent_messages = [
            msg for msg in state.messages 
            if msg["role"] in ["system", "user"]
        ]
        state.messages = persistent_messages
        
        # Add the new assistant response (which holds the current scratchpad state)
        state.add_message(
            role="assistant", 
            content=content, 
            tool_calls=tool_calls
        )

        try:
            # Basic cleanup for markdown code blocks (```json ... ```)
            clean_content = content.strip()
            if clean_content.startswith("```"):
                clean_content = clean_content.split("\n", 1)[-1].rsplit("\n", 1)[0]
            
            data = json.loads(clean_content)
            
            if data.get("finished") is True:
                state.is_finished = True
                logger.info(f"Agent finished. Report: {data.get('message')}")
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM. Continuing...")
        except Exception as e:
            logger.error(f"Error checking stop condition: {e}")

        if tool_calls:
            for tool_call in tool_calls:
                # This returns a dictionary: {'role': 'tool', ...}
                tool_msg = self.call_tool(tool_call)
                state.messages.append(tool_msg)
        
        return state