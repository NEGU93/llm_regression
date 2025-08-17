import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from src.model import Llama3Model
from src.evaluation import Tester


load_dotenv()


HF_USER = "NEGU93"
DATASET_NAME = f"{HF_USER}/pricer-data"
MAX_SEQUENCE_LENGTH = 182


hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)


if __name__ == "__main__":
    # Get model
    model = Llama3Model()
    BASE_MODEL = model.model_name

    # Get dataset
    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]
    test = dataset["test"]

    # Predict and evaluate
    model.predict(test[0]["text"])
    Tester.test(model.predict, test)
