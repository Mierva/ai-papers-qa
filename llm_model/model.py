from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from typing import Any, Dict, List, Optional
from pydantic import Field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.callbacks import CallbackManagerForLLMRun


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
        # Load tokenizer and explicitly set padding token
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

    def _generate(self, messages: List[str], max_length=100):
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model is not initialized. Call `_initialize_model()` first.")

        inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> AIMessage:
        """
        Generate a response based on the provided messages.
        """
        # Extract the text from the BaseMessage objects
        message_texts = [message.content for message in messages]

        # Generate a response
        response_text = self._generate(message_texts)

        # Apply stop sequences if provided
        if stop:
            for stop_token in stop:
                if stop_token in response_text:
                    response_text = response_text.split(stop_token)[0]
                    break

        # Return the response as an AIMessage
        return AIMessage(content=response_text)

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
