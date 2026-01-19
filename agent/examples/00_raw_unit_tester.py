"""
This is an Agent who given certain python files will write unit tests using pytest 
And will execute them and report result

Difference Between it And `01_raw_traced_unit_tester.py` no mlflow/langfuse 
"""
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_explorer_tools
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from llm.groq_client import GroqClient, LLMConfig
from loguru import logger
import json

from pathlib import Path

# ================ 1. Initalization ================
# 1.1 setup llm client
config = LLMConfig(
    max_tokens=5000,
    model_name="llama-3.3-70b-versatile", # Ensure using a valid model
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = GroqClient(config)

# 1.2 setup path
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

# ================ 2. Starts Iterations ================
max_iterations = 20
iteration = 0
while True:
    iteration += 1
    logger.info(f"Iteration {iteration}")
    
    client_tools = registry.to_client_tools(config.provider)
    response = client.generate(messages, tools=client_tools)
    
    messages.append(response)
    logger.info(f"Assistant Response: {response.get('content')}")

    content = response.get("content") or ""
    
    if iteration > max_iterations:
        logger.warning("Max iterations reached.")
        break

    if content:
        # Simple heuristic or try/except JSON parse to find "finished": true
        try:
            # Attempt to parse json if the model output pure json or markdown json
            clean_content = content.strip()
            if clean_content.startswith("```"):
                 clean_content = clean_content.split("\n", 1)[-1].rsplit("\n", 1)[0]
            
            data = json.loads(clean_content)
            if data.get("finished") is True:
                logger.success(f"Task Finished: {data.get('message')}")
                break
        except json.JSONDecodeError:
            pass # Continue if not valid json or not finished
    
    tool_calls = response.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call["type"] != "function":
            continue
        try:
            func_name = tool_call["function"]["name"]
            args_raw = tool_call["function"]["arguments"]
            
            if isinstance(args_raw, str):
                func_inputs = json.loads(args_raw)
            else:
                func_inputs = args_raw
            
            tool_instance = registry.get(func_name)
            if tool_instance:
                logger.debug(f"Executing tool {func_name}")
                func_results = tool_instance(**func_inputs)
            else:
                func_results = f"Error: Tool {func_name} not found."

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": str(func_results), # Content must be string
            }
        except Exception as error:
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": f"Error executing tool: {str(error)}",
            }
            
        messages.append(tool_message)
        logger.info(f"tool response {json.dumps(tool_message, indent=2)}")