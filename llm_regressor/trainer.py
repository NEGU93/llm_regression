from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from llm_regressor.model import Llama3Model
from llm_regressor import (
    lora_config,
    train_confifg,
    PROJECT_RUN_NAME,
    SEED,
    VALIDATION_PERCENTAGE,
)


def train(model: Llama3Model, train_dataset, eval_dataset=None):
    # Don't learn to predict the full description, just the price
    response_template = "Price is $"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=model.tokenizer
    )

    # Configs
    lora_parameters = LoraConfig(**lora_config)
    train_parameters = SFTConfig(**train_confifg)

    fine_tuning = SFTTrainer(
        model=model.base_model,
        train_dataset=train,
        eval_dataset=eval_dataset,
        peft_config=lora_parameters,
        args=train_parameters,
        data_collator=collator,
    )

    # Train
    fine_tuning.train()
    fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")


if __name__ == "__main__":
    from datasets import load_dataset
    from llm_regressor import DATASET_NAME

    # Get model
    model = Llama3Model()
    # Get dataset
    dataset = load_dataset(DATASET_NAME)
    train_data = dataset["train"]
    train_test_split = train_data.train_test_split(
        test_size=VALIDATION_PERCENTAGE, seed=SEED
    )
    train_split = train_test_split["train"]
    validation_split = train_test_split["test"]

    print(f"Training samples: {len(train_split)}")
    print(f"Validation samples: {len(validation_split)}")

    # Predict and evaluate
    train(model, train)
