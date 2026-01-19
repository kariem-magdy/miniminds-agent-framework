"""
This is an Agent who given certain python files will write unit tests using pytest 
And will execute them and report result
"""
import json
import os
from loguru import logger
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_explorer_tools
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from llm.groq_client import GroqClient, LLMConfig
from llm.config import LLMProvider
from pathlib import Path

# --- Robust Langfuse Setup ---
# We define the Dummy class first so it's always available
class DummyLangfuse:
    def trace(self, *args, **kwargs): return self
    def span(self, *args, **kwargs): return self
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, *args, **kwargs): pass

langfuse = None
traced_client_generate = None
traced_tool_execution = None

try:
    # Try importing and initializing Langfuse
    from langfuse import Langfuse, observe
    
    # Check for keys
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        langfuse = Langfuse()
        
        # Verify it actually works (has the trace method)
        if not hasattr(langfuse, 'trace'):
            raise AttributeError("Langfuse client missing 'trace' method")
            
        # If we got here, Langfuse is good. Use the real decorators.
        @observe(name="llm-call", as_type="generation")
        def real_traced_client_generate(client, messages, tools):
            return client.generate(messages, tools=tools)

        @observe(name="tool-call", as_type="tool")
        def real_traced_tool_execution(registry, tool_call):
            # (Same logic as before, just wrapped)
            return _execute_tool_logic(registry, tool_call)
            
        traced_client_generate = real_traced_client_generate
        traced_tool_execution = real_traced_tool_execution
        logger.info("Langfuse tracing enabled.")
    else:
        raise ValueError("Langfuse keys missing")

except Exception as e:
    logger.warning(f"Tracing disabled (Reason: {e}). Using dummy tracer.")
    langfuse = DummyLangfuse()
    
    # Define pass-through functions without @observe
    def traced_client_generate(client, messages, tools):
        return client.generate(messages, tools=tools)
        
    def traced_tool_execution(registry, tool_call):
        return _execute_tool_logic(registry, tool_call)

# Shared logic for tool execution (used by both traced and untraced)
def _execute_tool_logic(registry, tool_call):
    try:
        func_name = tool_call["function"]["name"]
        args_raw = tool_call["function"]["arguments"]
        func_inputs = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            
        tool_instance = registry.get(func_name)
        if not tool_instance:
             raise ValueError(f"Tool {func_name} not found")
             
        func_results = tool_instance(**func_inputs)

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": str(func_results),
        }
    except Exception as error:
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": f"Error: {str(error)}",
        }

# ================ 1. Initalization ================
config = LLMConfig(
    max_tokens=5000,
    model_name="llama-3.3-70b-versatile",
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = GroqClient(config)

files_under_test = ["tools/toolkit/web_explorer.py"]
tests_output_directory_path = "tools/llm_tests"

messages = [
    {
        "role": "system", "content": """
        You are a highly skilled QA Automation Agent with expertise in Python programming, unit testing (using Pytest), and modern GenAI tools. 
        Your role is to review, write, and execute comprehensive unit tests for the provided toolkit modules to ensure reliability and correctness. 
        Analyze each tool's behavior, suggest improvements if needed, and provide a clear, structured test report based on your findings.
        
        Output:
            - "finished": <boolean, indicate if the task is complete>
            - "message": <summary and coverage of tests>
        
        **Use only this tools:**
        {tools}
        you will be penalized if you use OTHER TOOLS
        """
    },
    {
        "role": "user", "content": """write unite tests for file: {files_under_test} code and run it ensure everything is okay then report it
        You are only allowed to change files in this directory {tests_output_directory_path}
        """ 
    }
]

registry = ToolRegistry()
registry.register_from_module(web_explorer_tools)
registry.register_from_module(code_tools)
registry.register_from_module(file_tools)
registry.register_from_module(json_tools)

messages[0]["content"] = messages[0]["content"].replace("{tools}", registry.to_string())
messages[1]["content"] = messages[1]["content"].format(
    files_under_test=str(files_under_test),
    tests_output_directory_path=str(tests_output_directory_path)
)

# Start Trace
root_span = langfuse.trace(name="unit-tester-run", metadata={"files": files_under_test})

# ================ 2. Starts Iterations ================
max_iterations = 20
iteration = 0
while True:
    iteration += 1
    logger.info(f"Iteration {iteration}")
    
    with root_span.span(name=f"iteration-{iteration}"):
        client_tools = registry.to_client_tools(config.provider)
        
        # 1. GENERATE
        response = traced_client_generate(client, messages, tools=client_tools)
        messages.append(response)
        
        # Safe Dictionary Access
        content = response.get('content') or ""
        tool_calls = response.get('tool_calls') or []
        
        logger.info(f"Assistant: {content[:100]}...")

        if iteration > max_iterations:
            break
            
        # Check finished
        if content:
            try:
                clean_content = content.strip()
                if clean_content.startswith("```"):
                     clean_content = clean_content.split("\n", 1)[-1].rsplit("\n", 1)[0]
                data = json.loads(clean_content)
                if data.get("finished") is True:
                    logger.success("Task Finished")
                    break
            except:
                pass
        
        # 2. EXECUTE TOOLS
        for tool_call in tool_calls:
            tool_msg = traced_tool_execution(registry, tool_call)
            messages.append(tool_msg)
            
# Close the trace
root_span.update(output=messages[-1])