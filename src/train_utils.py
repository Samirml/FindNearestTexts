import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.utils import compute_f1
from use_pretrained_model.extract_keywords import extract_keywords

def train(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, label_names: list) -> None:
    """
    Trains the model on the given dataset.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model weights.
        device (torch.device): Device to perform training on (CPU/GPU).
        label_names (list): List of NER label names.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    for i, batch in tqdm(enumerate(dataloader)):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids)

        loss = criterion(logits.permute(0, 2, 1), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        f1 = compute_f1(logits.detach().argmax(-1), labels, label_names)

        wandb.log({
            "train_loss": loss.item(),
            "train_f1": f1
        })


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, label_names: list) -> tuple:
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): DataLoader for the test data.
        device (torch.device): Device to perform evaluation on (CPU/GPU).
        label_names (list): List of NER label names.

    Returns:
        tuple: Test F1 score and average loss.
    """

    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    all_predictions = []
    all_labels = []
    for i, batch in tqdm(enumerate(dataloader)):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids)
        loss = criterion(logits.permute(0, 2, 1), labels)

        losses.append(loss.item())

        all_predictions.extend(logits.argmax(-1).cpu())
        all_labels.extend(labels.cpu())

    mean_loss = np.mean(losses)
    f1 = compute_f1(all_predictions, all_labels, label_names)

    wandb.log({
        "test_loss": mean_loss,
        "test_f1": f1
    })

    return f1, mean_loss

def save_train_texts_and_keywords(model, tokenizer, tokenized_datasets,device):
    texts = []
    keywords_per_text = []
    for i in tqdm(range(len(tokenized_datasets["train"]))):
        data = tokenized_datasets["train"][i]
        text = tokenizer.decode(data['input_ids'], skip_special_tokens=True)
        texts.append(text)
        keywords_per_text.append(set(extract_keywords(text, device, model)))
    texts = np.array(texts)
    return texts, keywords_per_text

def save_test_texts(model, tokenizer, tokenized_datasets):
    test_texts = []
    for i in tqdm(range(len(tokenized_datasets["test"]))):
        data = tokenized_datasets["test"][i]
        text = tokenizer.decode(data["input_ids"], skip_special_tokens = True)
        test_texts.append(text)
    return test_texts
