from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml


class LlamaModel:
    def __init__(self):
        with open('llm_model/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        self.model_weights = self.config['model_weights']

        self.tokenizer = None
        self.model = None

    def init(self):
        # Load tokenizer and explicitly set padding token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weights)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with VRAM optimization settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_weights,
            device_map="cuda",
            load_in_8bit=True,  # Enable 8-bit model loading
            torch_dtype=torch.float16,  # Use FP16 precision
            max_memory={  # Define maximum memory usage for CPU/GPU
                0: "10GiB",
                "cpu": "30GiB"
            }
        )

    def generate(self, prompt, max_length=100):
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model is not initialized. Call `init()` first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)