import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


def investigate_tokenizer(model_name):
    print("Investigating tokenizer for", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    for number in [0, 1, 10, 100, 999, 1000]:
        tokens = tokenizer.encode(str(number), add_special_tokens=False)
        print(f"The tokens for {number}: {tokens}")


def extract_price(s):
    if "Price is $" in s:
        contents = s.split("Price is $")[1]
        contents = contents.replace(",", "").replace("$", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0
    return 0


class Llama3Model:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B"):
        self.model_name = model_name

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.base_model.generation_config.pad_token_id = (
            self.tokenizer.pad_token_id
        )

        print(
            f"Memory footprint: {self.base_model.get_memory_footprint() / 1e9:.1f} GB"
        )

    def predict(self, prompt: str):
        set_seed(42)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.base_model.generate(
            inputs,
            max_new_tokens=4,
            attention_mask=attention_mask,
            num_return_sequences=1,
        )
        response = self.tokenizer.decode(outputs[0])
        return extract_price(response)
