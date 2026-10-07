import torch
import torch.nn as nn
from torch.optim import AdamW
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
import random
import os
import copy
import numpy as np

from .dataset import preprocess, prepare_dataloader


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module, unfreeze_pattern, verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        self.num_feat = nn.Sequential(
            nn.Linear(1, config.NUM_DIM),
            nn.ReLU(),
            nn.Linear(config.NUM_DIM, config.NUM_DIM)
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        self.hidden_dim = config.HIDDEN_DIM * 2 + config.NUM_DIM

        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, text_input, attention_mask, image_input, num_input):
        text_features = self.text_model(text_input, attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image_input)
        num_features = self.num_feat(num_input)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        comb_features = torch.cat([text_emb, image_emb, num_features], dim=1)
        
        result = self.regressor(comb_features)

        return result


def validate(model, val_loader, metric, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "text_input": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image_input": batch["image"].to(device),
                "num_input": batch["num_feat"].to(device)
            }
            label = batch["label"].to(device)

            pred = model(**inputs)
            loss = criterion(pred, label)
            val_loss += loss.item()
            _ = metric(preds=pred, target=label)
    val_mae = metric.compute().item()
    val_loss /= len(val_loader)
    metric.reset()
    return val_mae, val_loss
            


def train(conf, device):
    set_seed(conf.SEED)
    
    train, val = preprocess(conf, df_type="train")

    # model
    model = MultimodalModel(conf).to(device)
    tokenizer = AutoTokenizer.from_pretrained(conf.TEXT_MODEL_NAME)
    set_requires_grad(model.text_model, conf.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, conf.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': conf.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': conf.IMAGE_LR},
        {'params': model.num_feat.parameters(), 'lr': conf.NUM_LR},
        {'params': model.text_proj.parameters(), 'lr': conf.REGRESSOR_LR},
        {'params': model.image_proj.parameters(), 'lr': conf.REGRESSOR_LR},
        {'params': model.regressor.parameters(),'lr': conf.REGRESSOR_LR}
    ])

    criterion = nn.L1Loss()

    # Подготовка даталодеров
    train_dl = prepare_dataloader(conf, train, tokenizer, df_type="train")
    val_dl = prepare_dataloader(conf, val, tokenizer, df_type="val")

    # Метрика
    mae_train = torchmetrics.MeanAbsoluteError()
    mae_val = torchmetrics.MeanAbsoluteError()

    best_val_mae = float('inf')
    epochs_without_improvement = 0
    best_model_state = copy.deepcopy(model.state_dict())

    train_history = []

    for epoch in range(conf.EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_dl:
            inputs = {
                "text_input": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image_input": batch["image"].to(device),
                "num_input": batch["num_feat"].to(device)
            }
            label = batch["label"].to(device)

            optimizer.zero_grad()
            pred = model(**inputs)
            loss = criterion(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _ = mae_train(preds=pred, target=label)

        avg_epoch_loss = epoch_loss / len(train_dl)
        epoch_mae = mae_train.compute().item()
        val_mae, val_loss = validate(model, val_dl, mae_val, criterion, device)
        mae_train.reset()

        train_history.append({
            "epoch": epoch+1,
            "train_loss": avg_epoch_loss,
            "train_mae": epoch_mae,
            "val_mae": val_mae,
            "val_loss": val_loss
        })

        print(f"Epoch {epoch+1}/{conf.EPOCHS} | train_loss: {avg_epoch_loss:.4f} | "
              f" train_mae: {epoch_mae:.2f} | val_loss: {val_loss:.4f} | val_mae: {val_mae:.2f}")

        if val_mae < best_val_mae:
            print(f"New best model, epoch: {epoch+1}")
            best_val_mae = val_mae
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= conf.PATIENCE:
                print(f"Early stopping: {conf.PATIENCE} epochs without improvement")
                break
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), conf.SAVE_PATH)

    return train_history
            
            
    
