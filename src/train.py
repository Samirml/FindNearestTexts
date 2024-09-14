import torch
import wandb
import os
from torch.utils.data import DataLoader
from src.data import load_and_preprocess_data
import numpy as np
from src.model_upgrade import TorchLSTM
from src.train_utils import train, evaluate
from src.config import config
from use_pretrained_model.extract_keywords import extract_keywords


os.environ['TOKENIZERS_PARALLELISM']='false'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenized_datasets, tokenizer, ner_feature, label_names, data_collator = load_and_preprocess_data(config["tokenizer_name"])
train_loader = DataLoader(tokenized_datasets["train"], collate_fn = data_collator, shuffle=True, batch_size = 32)
test_loader = DataLoader(tokenized_datasets["test"], collate_fn = data_collator, shuffle=True, batch_size = 4)


model = TorchLSTM(vocab_size = len(tokenizer),
                  hidden_size = config["hidden_size"],
                  num_layers = config["num_layers"],
                  bidirectional=True,
                  n_classes=config["n_classes"]
).to(device)
print(config["tokenizer_name"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

wandb.init(project="text_recommendation", name="lstm")

for epoch in range(config["epochs"]):
    print(f"Epoch {epoch+1}/{config['epochs']}")
    train(model, train_loader, optimizer, device, label_names)
    test_f1, test_loss = evaluate(model, test_loader, device, label_names)

wandb.finish()

# Define the path where the model will be saved
save_path = 'model_versions/model_to_find_texts.pt'

# Create the parent directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the model state dictionary
torch.save(model.state_dict(), save_path)



