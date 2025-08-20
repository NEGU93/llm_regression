import os
from huggingface_hub import login
from dotenv import load_dotenv
from datetime import datetime

SEED = 42

# Get tokens
load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
wandb_api_key = os.getenv("WANDB_API_KEY", None)

# Huggin Face
login(hf_token, add_to_git_credential=True)
HF_USER = "NEGU93"
PROJECT_NAME = "pricer"

# Data
DATASET_NAME = f"{HF_USER}/pricer-data"
MAX_SEQUENCE_LENGTH = 182

# Run name for saving the model in the hub
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# wandb
# LOG_TO_WANDB = True if wandb_api_key else False
LOG_TO_WANDB = False  # Set to False for now, can be enabled later

# if wandb_api_key:
#     import wandb

#     wandb.login()
#     os.environ["WANDB_PROJECT"] = PROJECT_NAME
#     os.environ["WANDB_LOG_MODEL"] = "checkpoint"
#     os.environ["WANDB_WATCH"] = "gradients"

#     wandb.init(project=PROJECT_NAME, name=RUN_NAME)


# Hyperparameters for Training
EPOCHS = 1  # you can do more epochs if you wish, but only 1 is needed - more is probably overkill
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"
VALIDATION_PERCENTAGE = 0.1

# Admin config - note that SAVE_STEPS is how often it will upload to the hub
STEPS = 500
SAVE_STEPS = 5000

# Used for writing to output in color
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

# Confgiuration for Training
lora_config = {
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "r": 8,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

train_confifg = {
    "output_dir": PROJECT_RUN_NAME,
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "no",
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "optim": OPTIMIZER,
    "save_steps": SAVE_STEPS,
    "save_total_limit": 10,
    "logging_steps": STEPS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": WARMUP_RATIO,
    "group_by_length": True,
    "lr_scheduler_type": LR_SCHEDULER_TYPE,
    "report_to": "wandb" if LOG_TO_WANDB else None,
    "run_name": RUN_NAME,
    "max_seq_length": MAX_SEQUENCE_LENGTH,
    "dataset_text_field": "text",
    "save_strategy": "steps",
    "hub_strategy": "every_save",
    "push_to_hub": True,
    "hub_model_id": HUB_MODEL_NAME,
    "hub_private_repo": True,
}
