import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    
    provider: LLMProvider = Field(default=LLMProvider.GROQ, description="The LLM provider to use")
    
    base_url: Optional[str] = Field( default=None, description="API Base URL (only needed for custom endpoints)")

    model_name: str = Field( default="llama-3.1-70b-versatile", description="model name")
    
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature") 
       
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    
    max_tokens: int = Field(default=4096, description="Maximum number of tokens to generate")    
    
    reasoning_effort: str = Field(default="medium", description="Effort level for reasoning models (e.g. o1/o3)")