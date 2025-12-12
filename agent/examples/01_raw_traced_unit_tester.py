"""
This is an Agent who given certain python files will write unit tests using pytest 
And will execute them and report result

Difference Between it And `00_raw_traced_unit_tester.py` no mlflow/langfuse 
"""
import json
from loguru import logger
from langfuse import observe, get_client
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_explorer_tools
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from llm.groq_client import GroqClient, LLMConfig
from llm.config import LLMProvider
from pathlib import Path


langfuse = get_client()

@observe(name="llm-call", as_type="generation")
def traced_client_generate(client, messages, tools):
    #TODO: call client (provide tools in .generate) with registery.to_client_tools()
    return client.generate(messages, tools)


@observe(name="tool-call", as_type="tool")
def traced_tool_execution(registery, tool_call):
    try:
        # TODO: extract func_name, args and call it -> set tool_message {content: result}
        func_name = tool_call["function"]["name"]
        args_raw = tool_call["function"]["arguments"]
        if isinstance(args_raw, str):
            func_inputs = json.loads(args_raw)
        else:
            func_inputs = args_raw
        
        # Call the tool from registry
        tool = registery.get(func_name)
        func_results = tool(**func_inputs)

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": json.dumps(func_results),
        }
        return tool_message
    except Exception as error:
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": json.dumps({"success": False, "error": str(error)}),
        }
                
# ================ 1. Initalization ================
# 1.1 setup llm client
config = LLMConfig(
    max_tokens=5000,
    model_name="openai/gpt-oss-120b",
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = GroqClient(config)

# 1.2 setup path
messages = [
    {
        "role": "system", "content": """
        You are a highly skilled QA Automation Agent with expertise in Python programming, unit testing (using Pytest), and modern GenAI tools. 
        Your role is to review, write, and execute comprehensive unit tests for the provided toolkit modules to ensure reliability and correctness. 
        Analyze each tool's behavior, suggest improvements if needed, and provide a clear, structured test report based on your findings.
        
        Output:
            - "finished": <boolean, indicate if the task is complete>
            - "message": <summary and coverage of tests>
        
        **CRITICAL: ONLY use the tools listed below. Do NOT use any other tool names like repo_browser, file_browser, or any tools not explicitly listed here:**
        {tools}
        
        **Any attempt to use tools not in the above list will fail. Only call functions that are explicitly provided above.**
        """
    },
    {
        "role": "user", "content": """write unite tests for file: {files_under_test} code and run it ensure everything is okay then report it
        You are only allowed to change files in this directory {tests_output_directory_path}
        """ 
    }
]
# TODO: 1.3 create tool register and add tools/modules you need
registery = ToolRegistry("unit_tester_traced_session")
registery.register_from_module(code_tools)
registery.register_from_module(file_tools)
registery.register_from_module(json_tools)

# TODO: 1.4 add tools to system_message use str.format method and  registery.to_string()
messages[0]["content"] = messages[0]["content"].format(tools=registery.to_string())

# TODO: 1.5 set `files_under_test` and `tests_output_directory_path` in user_message like .format
files_under_test = ["tools/toolkit/builtin/json_tools.py"]
tests_output_directory_path = "tests/unit_tests"
messages[1]["content"] = messages[1]["content"].format(
    files_under_test=", ".join(files_under_test),
    tests_output_directory_path=tests_output_directory_path
)

# TODO 1.6 set root span with root_span = langfuse.start_span(name , metadata)
root_span = langfuse.start_span(
    name="unit-tester-agent",
    metadata={
        "files_under_test": files_under_test,
        "output_directory": tests_output_directory_path,
        "model": config.model_name
    }
)

# ================ 2. Starts Iterations ================
max_iterations = 20
iteration = 0
while True:
    iteration += 1
    logger.info(f"Iteration {iteration}")
    with root_span.start_as_current_observation(as_type="span", name=f"iteration-{iteration}"):
        # TODO 2.1 call client (provide tools in .generate) with registery.to_client_tools()
        try:
            response = traced_client_generate(client, messages, tools=registery.to_client_tools(config.provider))
        except Exception as e:
            # Handle API errors (like invalid tool calls)
            error_message = str(e)
            logger.error(f"API error: {error_message}")
            
            # If the error is about invalid tool names, provide feedback
            if "attempted to call tool" in error_message:
                available_tools = ", ".join([tool.name for tool in registery._tools.values()])
                feedback_msg = {
                    "role": "user",
                    "content": f"ERROR: You tried to use a tool that doesn't exist. Available tools are ONLY: {available_tools}. Do NOT prefix tool names with 'repo_browser.' or any other prefix. Use the exact tool names listed."
                }
                messages.append(feedback_msg)
                logger.info(f"Sent error feedback to agent: {feedback_msg['content']}")
                continue
            else:
                raise  # Re-raise if it's a different error
        
        # TODO 2.2 append assistant message (role, content, **tool_calls**) *log it logger.info*
        assistant_message = {
            "role": "assistant",
            "content": response.content,
        }
        # Only add tool_calls if they exist (Groq doesn't accept null tool_calls)
        if response.tool_calls:
            assistant_message["tool_calls"] = response.tool_calls
        
        messages.append(assistant_message)
        logger.info(f"Assistant response: {json.dumps(assistant_message, indent=2, default=str)}")

        # TODO get content and check if is finished
        # 2.3 Stop when one of the conditions happen
        # 2.3 'finished' in response['content'] -- handle response['content']=None case
        # 2.3 exceed max_iterations
        content = response.content or ""
        if "finished" in content.lower() or iteration >= max_iterations:
            logger.info(f"Agent finished. Final message: {content}")
            break
        
        # 2.4 execute any function execturion inside `tool_calls` || handle if it's None or not passed
        tool_calls = response.tool_calls or []
        for tool_call in tool_calls:
            if tool_call["type"] != "function":
                continue
            
            tool_message = traced_tool_execution(registery, tool_call)
                
            messages.append(tool_message)
            logger.info(f"tool response {json.dumps(tool_message, indent=2)}")
            
root_span.end()