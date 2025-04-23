# library imports
import os
import json
import torch
import torch.nn as nn
import numpy as np
import csv
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from datasets import load_dataset
import evaluate

# Config class for Mamba
@dataclass
class MambaConfig:
    d_model: int = 768
    n_layer: int = 24
    d_intermediate: int = 3072
    vocab_size: int = 50280
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def to_json_string(self):
        return json.dumps(asdict(self), indent=2)

    def to_dict(self):
        return asdict(self)

# Classification head
class MambaClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(d_model, num_classes, **kwargs)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        return self.classification_head(hidden_states)

# Mamba Model for Classification
class MambaTextClassification(MambaLMHeadModel):
    def __init__(self, config: MambaConfig, num_classes=2, initializer_cfg=None, device=None, dtype=None):
        super().__init__(config, initializer_cfg, device, dtype)
        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=num_classes)
        del self.lm_head
        self.backbone.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.backbone(input_ids)
        mean_hidden_states = hidden_states.mean(dim=1)
        logits = self.classification_head(mean_hidden_states)
        if labels is None:
            return namedtuple("ClassificationOutput", ["logits"])(logits=logits)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return namedtuple("ClassificationOutput", ["loss", "logits"])(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        model.load_state_dict(state_dict, strict=False)
        return model

# Dataset
DATASET = "imdb"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

def preprocess_function(texts):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

if DATASET == "imdb":
    raw_data = load_dataset("imdb")
    train_texts, train_labels = raw_data["train"]["text"], raw_data["train"]["label"]
    test_texts, test_labels = raw_data["test"]["text"], raw_data["test"]["label"]
    num_classes = 2
else:
    raise NotImplementedError("Only IMDB dataset is currently supported.")

tokenized_train = preprocess_function(train_texts)
tokenized_test = preprocess_function(test_texts)

class TextDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_data.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextDataset(tokenized_train, train_labels)
test_dataset = TextDataset(tokenized_test, test_labels)

# Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)
   

# custom CSV Logger
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_path="eval_results.csv"):
        self.output_path = output_path
        self.fields = ["step", "eval_loss", "eval_accuracy", "training_loss"]
        self.latest_train_loss = None

        with open(self.output_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.latest_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        row = {
            "step": state.global_step,
            "eval_loss": metrics.get("eval_loss"),
            "eval_accuracy": metrics.get("eval_accuracy"),
            "training_loss": self.latest_train_loss if self.latest_train_loss is not None else "NA"
        }
        with open(self.output_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)


# custom Trainer
class MambaTrainer(Trainer):
    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(output_dir)
        with open(f'{output_dir}/mamba_config.json', 'w') as f:
            f.write(self.model.config.to_json_string())

# Model initilization
model = MambaTextClassification.from_pretrained(
    "state-spaces/mamba-130m",
    device="cuda",
    num_classes=num_classes
).to("cuda")

with torch.no_grad():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Training arguments
training_args = TrainingArguments(
    output_dir="mamba_model",
    report_to="none",
    learning_rate=5e-5,
    weight_decay=0.03,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_strategy="steps",
    logging_steps=100,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    gradient_accumulation_steps=4,
    fp16=False,
    bf16=True,
    max_grad_norm=0.5,
)

trainer = MambaTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[CSVLoggerCallback("eval_results.csv")]
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True, max_split_size_mb:32"

# train
trainer.train()

# cm plot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(preds, labels, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

final_output = trainer.predict(test_dataset)
preds = np.argmax(final_output.predictions, axis=1)
labels = final_output.label_ids
plot_confusion_matrix(preds, labels, class_names=["Negative", "Positive"])
