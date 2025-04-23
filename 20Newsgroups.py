# import necessary libraries
import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
import evaluate
from accelerate import Accelerator

# Configs for mamba model
@dataclass
class MambaConfig:
    d_model: int = 768
    n_layer: int = 24
    d_intermediate: int = 3072
    vocab_size: int = 50257
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def to_json_string(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)

# Classification head
class MambaClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.2, **kwargs):
        super(MambaClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(d_model, num_classes, **kwargs)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        return self.classification_head(hidden_states)

# Mamba model for classificataion
class MambaTextClassification(MambaLMHeadModel):
    def __init__(self, config: MambaConfig, initializer_cfg=None, device=None, dtype=None):
        super().__init__(config, initializer_cfg, device, dtype)
        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=20)
        del self.lm_head
        self.backbone.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.backbone(input_ids)
        mean_hidden_states = hidden_states.mean(dim=1)
        logits = self.classification_head(mean_hidden_states)
        if labels is None:
            ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOutput(logits=logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return ClassificationOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        model.load_state_dict(model_state_dict, strict=False)
        return model

# Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Tokenization and Preprocessing
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = MambaTextClassification.from_pretrained("state-spaces/mamba-130m", device="cuda")
model.to("cuda")

for name, param in model.named_parameters():
    print(name, param.shape)

newsgroups = fetch_20newsgroups(subset='all')
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')
train_texts, train_labels = train_data.data, train_data.target
test_texts, test_labels = test_data.data, test_data.target

def preprocess_function(texts):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512) 

tokenized_train = preprocess_function(train_texts)
tokenized_test = preprocess_function(test_texts)
 
class NewsgroupsDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_data.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = NewsgroupsDataset(tokenized_train, train_labels)
test_dataset = NewsgroupsDataset(tokenized_test, test_labels)

# custom Trainer
class MambaTrainer(Trainer):
    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer 
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(output_dir)
        with open(f'{output_dir}/config.json', 'w') as f:
            json.dump(self.model.config.to_dict(), f)

# Training arguments
training_args = TrainingArguments(
    output_dir="mamba_model",
    report_to="none",
    learning_rate=2e-5,
    weight_decay=0.03,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=15,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine_with_restarts",
    eval_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    save_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    push_to_hub=False,
    load_best_model_at_end=False,
    gradient_accumulation_steps=16,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    tokenizer=tokenizer
)

accelerator = Accelerator()
model = accelerator.prepare(model)

with torch.no_grad():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True, max_split_size_mb:32"

trainer.train()
