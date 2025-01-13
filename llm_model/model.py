from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import torch
import yaml


@dataclass
class Args:
    pass


@dataclass
class Result:
    pass


class LlamaModel:
    def __init__(self):
        # Load configuration
        with open('D:\Projects\MLOPS_proj\llm_model\config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Define model and tokenizer paths
        self.model_weights = self.config['model_weights']

        # Defer model and tokenizer initialization to save VRAM
        self.tokenizer = None
        self.model = None

    def init(self):
        # Load tokenizer and explicitly set padding token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weights)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure pad_token is set to eos_token

        # Load model with VRAM optimization settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_weights,
            device_map="auto",  # Automatically allocate layers to CPU/GPU
            load_in_8bit=True,  # Enable 8-bit model loading
            torch_dtype=torch.float16,  # Use FP16 precision
            max_memory={  # Define maximum memory usage for CPU/GPU
                0: "10GiB",  # Adjust this to fit your GPU's memory
                "cpu": "30GiB"
            }
        )

    def generate(self, prompt, max_length=100):
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model is not initialized. Call `init()` first.")

        # Encode input and move to the model's device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate output
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length)

        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
# llama = LlamaModel()
# print(f'config: {llama.config}')
# print(f'weights: {llama.model_weights}')

# llama.init()
# output = llama.generate(prompt="hi, can you say 'hi, orichumaru' to me?")
# print(f'output: {output}')
