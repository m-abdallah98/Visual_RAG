### Models
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from llama_index.llms.gemini import Gemini
from llama_index.llms.cohere import Cohere
from llama_index.llms.cohere.utils import CHAT_MODELS, ALL_AVAILABLE_MODELS, COMMAND_MODELS
from llama_index.llms.ollama import Ollama

## Add Aya-Expanse manually to CHAT_MODELS as it is not officially supported yet
if "c4ai-aya-expanse-32b" not in CHAT_MODELS:
    CHAT_MODELS["c4ai-aya-expanse-32b"] = 128000
    ALL_AVAILABLE_MODELS["c4ai-aya-expanse-32b"] = 128000
    COMMAND_MODELS["c4ai-aya-expanse-32b"] = 128000

## Set model
def get_llm(model):
    if model == "Llama":
        return TogetherLLM(model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo")
    elif model == "Qwen":
        return TogetherLLM(model="Qwen/Qwen2.5-72B-Instruct-Turbo")
    elif model == "GPT":
        return OpenAI(model="gpt-4o-mini")
    elif model == "Gemini":
        return Gemini(model="models/gemini-1.5-pro")
    elif model == "Aya-Expanse":
        return Cohere(model="c4ai-aya-expanse-32b")
    elif model == "Ollama":
        return Ollama(model="llama3.2:latest", request_timeout=120.0)
    else:
        raise ValueError("Invalid model")
