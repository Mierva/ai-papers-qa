from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from typing import Any, Dict, Optional
from pydantic import Field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult


class LlamaModel(BaseChatModel):
    model_weights: Optional[str] = Field(None, description="Path to the model weights.")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)

    def __init__(self, config_path: str = 'llm_model/config.yaml', **kwargs):
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Initialize required attributes
        kwargs['model_weights'] = config.get('model_weights')
        if not kwargs['model_weights']:
            raise ValueError("Model weights must be specified in the configuration.")

        super().__init__(**kwargs)

        # Set tokenizer and model to None (initialized later)
        self.tokenizer = None
        self.model = None

    def _initialize_model(self):
        if not self.model_weights:
            raise ValueError("Model weights are not initialized.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weights)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with VRAM optimization settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_weights,
            device_map="cuda",
            load_in_8bit=True,  # Enable 8-bit model loading
            torch_dtype=torch.float16,  # Use FP16 precision
            max_memory={
                0: "10GiB",
                "cpu": "30GiB"
            }
        )

    def _generate(self,
                  messages: list[BaseMessage] | list[HumanMessage],
                  stop: Optional[list[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  max_length=150) -> ChatResult:
        # Extract text from HumanMessage if necessary
        if isinstance(messages[0], HumanMessage):
            messages = [message.content for message in messages]

        print(f"Messages to generate: {messages}")
        inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)
        # outputs = self.model.generate(inputs["input_ids"], max_length=max_length)
        outputs = self.model.generate(
            messages,
            return_tensors="pt",
            padding=True,
            #TODO: write a function to get the max_length depending on the user's request
            max_new_tokens=50,  # Limit the generation to 50 new tokens
            temperature=0.7,    # Optional: Adjust creativity
            do_sample=True,     # Optional: Enable sampling
            top_p=0.95,         # Optional: Nucleus sampling for diversity
        ).to(self.model.device)

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        ct_input_tokens = sum(len(message) for message in messages)
        ct_output_tokens = len(decoded_output)

        message = AIMessage(
            content=decoded_output,
            additional_kwargs={},
            response_metadata={
                "time_in_seconds": 3,  # Placeholder for actual time
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _call(self, messages: list[BaseMessage] | list[HumanMessage]) -> ChatResult:
        return self._generate(messages)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "Llama3.1B-8bit quantized"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_weights": self.model_weights,
        }