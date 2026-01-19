import os
from typing import Iterator, List, Optional
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from .base import LLMClient
from .config import LLMConfig

# 1: load dotenv
load_dotenv()

class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # 2: create groq client and set api_key from .env
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
             # Fallback/Warning if needed
             pass
        self.client = Groq(api_key=api_key)
    
    def generate(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        """ 
        Send messages to Groq and return the full response dict.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
        except BadRequestError as e:
            # Handle specific Groq tool use errors (prevent crash)
            error_body = getattr(e, 'body', {})
            error_details = error_body.get('error', {})
            if error_details.get('code') == 'tool_use_failed':
                return {
                    "role": "assistant",
                    "content": f"System Error: The model failed to generate a valid tool call. Raw Error: {error_details.get('message')}"
                }
            # Re-raise other errors (like invalid API key)
            raise e
        
        # Extract only necessary fields to avoid sending unsupported metadata back to API
        msg_obj = response.choices[0].message
        
        response_dict = {
            "role": msg_obj.role,
            "content": msg_obj.content,
        }
        
        if msg_obj.tool_calls:
             response_dict["tool_calls"] = [
                 t.model_dump() for t in msg_obj.tool_calls
             ]
             
        return response_dict
    
    def stream(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Iterator[dict]:
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            stream=True
        )
        for chunk in stream:
            if chunk.choices:
                yield chunk.choices[0].delta.model_dump()

if __name__ == "__main__":
    # Test block
    config = LLMConfig(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024
    )
    try:
        client = GroqClient(config)
        print("GroqClient initialized successfully.")
    except Exception as e:
        print(f"Skipping execution: {e}")