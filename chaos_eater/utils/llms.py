import os
import os
import requests
from typing import List, Tuple, Callable, Iterator
import json
import tiktoken
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from .exceptions import ModelNotFoundError
from .wrappers import LLM, LLMBaseModel, BaseModel
from typing import Union, Any, Dict, Optional


def safe_get_response_field(response: Union[Dict, str, Any], field_name: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    Safely extract a field from an LLM response that could be either a dictionary or string.
    
    This utility handles the different response formats from various model providers:
    - OpenAI: Returns structured dict responses
    - GitHub AI: May return JSON strings or dict objects
    - Anthropic: Returns structured responses  
    - Google: Returns structured responses
    - Ollama: Returns structured responses
    
    Args:
        response: The response object from the LLM (dict, str, or other)
        field_name: The field name to extract
        fallback: Default value if field not found or response is invalid
        
    Returns:
        The extracted field value as string, or fallback if not found
    """
    # Handle None response
    if response is None:
        return fallback
    
    # Handle dictionary response (normal case)
    if isinstance(response, dict):
        return response.get(field_name, fallback)
    
    # Handle string response (some providers return JSON strings)
    if isinstance(response, str):
        try:
            import json
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed.get(field_name, fallback)
        except (json.JSONDecodeError, ValueError):
            # If it's just a plain string, return it if field_name matches common patterns
            if field_name in ['report', 'summary', 'content', 'text', 'message']:
                return response
        return fallback
    
    # Handle other types by converting to string
    return str(response) if fallback is None else fallback


def validate_llm_response_field(response: Union[Dict, str, Any], field_name: str, error_message: Optional[str] = None) -> str:
    """
    Validate and extract a required field from an LLM response with comprehensive error handling.
    
    Args:
        response: The response object from the LLM
        field_name: The required field name to extract
        error_message: Custom error message if extraction fails
        
    Returns:
        The validated field value as a non-empty string
        
    Raises:
        ValueError: If the field cannot be extracted or is empty
    """
    if error_message is None:
        error_message = f"Failed to extract '{field_name}' from LLM response"
    
    # Extract the field using safe_get_response_field
    field_value = safe_get_response_field(response, field_name)
    
    # Validate the extracted value
    if field_value is None:
        raise ValueError(f"{error_message}: Field '{field_name}' not found in response")
    
    if not isinstance(field_value, str):
        field_value = str(field_value)
    
    if not field_value.strip():
        raise ValueError(f"{error_message}: Field '{field_name}' is empty")
    
    return field_value


def safe_stream_response_extract(response: Union[Dict, str, Any], field_name: str, default_error_msg: Optional[str] = None) -> str:
    """
    Safely extract a field from streaming LLM responses with fallback error messages.
    
    This is specifically designed for streaming responses where the final response
    might be incomplete or in an unexpected format.
    
    Args:
        response: The final response token from streaming
        field_name: The field name to extract
        default_error_msg: Default error message if extraction fails
        
    Returns:
        The extracted field value or a meaningful error message
    """
    if default_error_msg is None:
        default_error_msg = f"Analysis failed: No {field_name} generated from the LLM response."
    
    # Try to extract the field
    field_value = safe_get_response_field(response, field_name)
    
    # Handle None or empty responses
    if field_value is None:
        return default_error_msg
    
    # Ensure we have a string
    if not isinstance(field_value, str):
        field_value = str(field_value)
    
    # Handle empty strings
    if not field_value.strip():
        return f"Analysis failed: Empty {field_name} generated from the LLM response."
    
    return field_value


def verify_model_name(model_name: str) -> bool:
    resp = requests.get(f"http://localhost:11434/api/tags")
    resp.raise_for_status()
    data = resp.json()
    return any(model.get("name") == model_name for model in data.get("models", []))


def load_llm(
    model_name: str,
    temperature: float = 0.0,
    port: int = 8000,
    seed: int = 42,
    github_base_url: str = "https://models.github.ai/inference"
) -> LLM:
    if model_name.startswith("openai/"):
        return ChatOpenAI(
            model=model_name.split("openai/", 1)[1],
            temperature=temperature,
            seed=seed,
            request_timeout=30.0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    elif model_name.startswith("google/"):
        return ChatGoogleGenerativeAI(
            model=model_name.split("google/", 1)[1],
            temperature=temperature,
            model_kwargs={"generation_config": {"response_mime_type": "application/json"}}
        )
    elif model_name.startswith("anthropic/"):
        return ChatAnthropic(
            model=model_name.split("anthropic/", 1)[1],
            temperature=temperature,
            max_tokens=8192
            # model_kwargs=model_kwargs
        )
    elif model_name.startswith("github/"):
        return ChatOpenAI(
            model=model_name.split("github/", 1)[1],
            temperature=temperature,
            seed=seed,
            request_timeout=30.0,
            openai_api_key=os.environ.get("GITHUB_TOKEN"),
            openai_api_base=github_base_url,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    elif model_name.startswith("ollama/"):
        model_exists = verify_model_name(model_name.split("ollama/", 1)[1])
        if model_exists:
            return ChatOllama(
                model=model_name.split("ollama/", 1)[1],
                temperature=temperature,
                seed=seed
            )
        else:
            raise ModelNotFoundError(f"{model_name} was not found in the available list.")
    else:
        # Note: VLLMOpenAI is for base models
        #       ref: https://python.langchain.com/v0.2/docs/integrations/chat/vllm/
        return ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",
            openai_api_base=f"http://localhost:{port}/v1",
            temperature=temperature,
            max_tokens=2048,
            model_kwargs={"seed": seed}
        )


def build_json_agent(
    llm: LLM,
    chat_messages: List[Tuple[str, str]],
    pydantic_object: LLMBaseModel,
    is_async: bool = False,
    enables_prefill: bool = True,
    streaming_func: Callable = None
) -> Runnable:
    if enables_prefill:
        first_key = str(list(pydantic_object.__fields__.keys())[0])
        prefill_str = '```json\n{{\"{key}\":'.replace("{key}", first_key)
        # chat_messages.append(("human", 'The keys and values in the output JSON dictionary must always be enclosed in single double quotes. Triple quotes (""" or ```) must not be used.'))
        chat_messages.append(("ai", prefill_str)) # add json prefill
        # chat_messages.append(("human", "Please continue the output from where it left off."))
        # chat_messages.append(("human", "Please continue the subsequent output from the middle."))
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    prompt = ChatPromptTemplate.from_messages(chat_messages)
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    if streaming_func is None:
        if is_async:
            async def extract_json_items_streaming(input_stream):
                async for input in input_stream:
                    if not isinstance(input, dict):
                        continue
                    yield {key: input.get(key) for key in pydantic_object.__fields__.keys()}
        else:
            def extract_json_items_streaming(input_stream):
                for input in input_stream:
                    if not isinstance(input, dict):
                        continue
                    yield {key: input.get(key) for key in pydantic_object.__fields__.keys()}
    else:
        extract_json_items_streaming = streaming_func
    if enables_prefill:
        if is_async:
            async def add_prefill(input):
                buffer = ""
                prefix_added = False
                prefill_len = len(prefill_str) + 5 # margin
                async for chunk in input:
                    if not prefix_added:
                        buffer += chunk.content
                        buffer_len = len(buffer)
                        if buffer_len >= prefill_len:
                            if "```json" in buffer:
                                yield buffer
                                prefix_added = True
                            else:
                                prefill_str_ = prefill_str.replace("{{", "{")
                                if prefill_str_.replace("```json\n", "") in buffer.replace("\n", ""):
                                    yield "```json\n" + buffer
                                else:
                                    yield prefill_str_ + buffer
                                prefix_added = True
                    else:
                        yield chunk.content
        else:
            def add_prefill(input: Iterator[str]) -> Iterator[str]:
                buffer = ""
                prefix_added = False
                prefill_len = len(prefill_str) + 5 # margin
                for chunk in input:
                    if not prefix_added:
                        buffer += chunk.content
                        buffer_len = len(buffer)
                        if buffer_len >= prefill_len:
                            if "```json" in buffer:
                                yield buffer
                                prefix_added = True
                            else:
                                prefill_str_ = prefill_str.replace("{{", "{")
                                if prefill_str_.replace("```json\n", "") in buffer.replace("\n", ""):
                                    yield "```json\n" + buffer
                                else:
                                    yield prefill_str_ + buffer
                                prefix_added = True
                    else:
                        yield chunk.content
        agent = prompt | llm | add_prefill | parser | extract_json_items_streaming
    else:
        agent = prompt | llm | parser | extract_json_items_streaming
    return agent


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int

class LLMLog(BaseModel):
    name: str
    token_usage: TokenUsage
    message_history: List[List[str] | str]

class LoggingCallback(BaseCallbackHandler):
    def __init__(
        self,
        name: str,
        llm: LLM,
        streaming: bool = True
    ) -> None:
        if "model" in list(llm.__fields__.keys()):
            self.model_name = llm.model
            if "gemini" in self.model_name:
                self.model_provider = "google"
            elif "claude" in self.model_name:
                self.model_provider = "anthropic"
            elif "gpt" in self.model_name:
                self.model_provider = "github" if "GITHUB_TOKEN" in os.environ and "github.ai" in getattr(llm, 'openai_api_base', '') else "openai"
            else:
                # raise TypeError(f"Invalid model name: {self.model_name}") # TODO: Implement more appropriate error handling.
                self.model_provider = "ollama"
        elif "model_name" in list(llm.__fields__.keys()):
            self.model_name = llm.model_name
            if "gpt" in self.model_name:
                self.model_provider = "github" if "GITHUB_TOKEN" in os.environ and "github.ai" in getattr(llm, 'openai_api_base', '') else "openai"
            else:
                # raise TypeError(f"Invalid model name: {self.model_name}") # TODO: Implement more appropriate error handling.
                self.model_provider = "ollama"
        else:
            raise TypeError(f"Invalid llm: {llm}")
        self.streaming = streaming
        if self.model_provider in ["openai", "github"] and self.streaming:
            self.enc = tiktoken.encoding_for_model(self.model_name)
        self.token_usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0
        )
        self.message_history = []
        self.name = name
        self.log = LLMLog(
            name=self.name,
            token_usage=self.token_usage,
            message_history=self.message_history
        )

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.message_history.append(prompts)
        if self.model_provider in ["openai", "github"] and self.streaming:
            for prompt in prompts:
                self.token_usage.input_tokens += len(self.enc.encode(prompt))

    def on_llm_end(self, response: LLMResult, **kwargs):
        for generations in response.generations:
            for generation in generations:
                if self.model_provider in ["openai", "github"] and self.streaming:
                    self.token_usage.output_tokens += len(self.enc.encode(generation.text))
                    self.token_usage.total_tokens = self.token_usage.input_tokens + self.token_usage.output_tokens
                else:
                    if self.model_provider in ["openai", "github"]:
                        tokens = generation.message.response_metadata.get("token_usage")
                        self.token_usage.input_tokens += tokens.get("prompt_tokens", -1)
                        self.token_usage.output_tokens += tokens.get("completion_tokens", -1)
                        self.token_usage.total_tokens += tokens.get("total_tokens", -1)
                    elif self.model_provider in ["google", "anthropic"]:
                        tokens = generation.message.usage_metadata
                        self.token_usage.input_tokens += tokens.get("input_tokens", -1)
                        self.token_usage.output_tokens += tokens.get("output_tokens", -1)
                        self.token_usage.total_tokens += tokens.get("total_tokens", -1)
                self.message_history.append(generation.text)
        self.log = LLMLog(
            name=self.name,
            token_usage=self.token_usage,
            message_history=self.message_history
        )

UNIT = 1e+6
PRICING_PER_TOKEN = {
    "openai/gpt-4o-2024-08-06": {
        "input": 2.50 / UNIT,
        "output": 10. / UNIT
    },
    "openai/gpt-4o-2024-05-13": {
        "input": 5.00 / UNIT,
        "output": 15.00 / UNIT
    },
    "openai/gpt-4o-mini-2024-07-18": {
        "input": 0.15 / UNIT,
        "output": 0.6 / UNIT
    },
    "google/gemini-2.0-flash-lite": {
        "input": 3.50 / UNIT,
        "output": 10.50 / UNIT
    },
    "google/gemini-1.5-pro": {
        "input": 3.50 / UNIT,
        "output": 10.50 / UNIT
    },
    "anthropic/claude-3-5-sonnet-20241022": {
        "input": 3.75 / UNIT,
        "output": 15. / UNIT
    },
    "anthropic/claude-3-5-sonnet-20240620": {
        "input": 3.75 / UNIT,
        "output": 15. / UNIT
    },
    "github/gpt-4o": {
        "input": 0.0 / UNIT,  # GitHub AI models are free for now
        "output": 0.0 / UNIT
    },
    "github/gpt-4o-mini": {
        "input": 0.0 / UNIT,  # GitHub AI models are free for now
        "output": 0.0 / UNIT
    },
}

def verify_api_key(provider: str, api_key: str) -> bool:
    if not api_key:
        return False
    try:
        if provider == "openai":
            client = OpenAI(api_key=api_key)
            try:
                client.models.list()
            except openai.AuthenticationError:
                return False
            else:
                return True
        elif provider == "github":
            client = OpenAI(
                api_key=api_key,
                base_url="https://models.github.ai/inference"
            )
            try:
                client.models.list()
            except openai.AuthenticationError:
                return False
            else:
                return True
        elif provider == "anthropic":
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        elif provider == "google":
            response = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                timeout=5
            )
            return response.status_code == 200
    except Exception:
        return False
    return False

def get_env_key_name(provider: str) -> str:
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "github": "GITHUB_TOKEN"
    }
    return env_keys.get(provider, "API_KEY")

def check_existing_key(provider: str) -> bool:
    env_key_name = get_env_key_name(provider)
    existing_key = os.environ.get(env_key_name)
    if existing_key:
        if verify_api_key(provider, existing_key):
            return True
    return False