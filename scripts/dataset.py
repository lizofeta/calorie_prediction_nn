import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functools import partial
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import torch
import re


class MultimodalDataset(Dataset):
    def __init__(self, df, conf, transforms):
        self.df = df
        self.conf = conf
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "ingredients"]
        label = self.df.loc[idx, "total_calories"]
        num_feature = self.df.loc[idx, "total_mass"]
        
        dish_id = self.df.loc[idx, "dish_id"]
        img_path = f"{self.conf.IMAGE_ROOT}/{dish_id}/{self.conf.IMAGE_NAME}"
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(image=np.array(img))["image"]

        return {
            "text": text,
            "image": img,
            "num_feat": num_feature,
            "label": label
        }


def collate_fn(batch, tokenizer):
    text = [item["text"] for item in batch]
    label = torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1)
    img = torch.stack([item["image"] for item in batch])
    num_feat = torch.tensor([item["num_feat"] for item in batch], dtype=torch.float32).unsqueeze(1)

    tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)

    return {
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "image": img,
        "num_feat": num_feat,
        "label": label
    }


def move_ingredients(ingredients, id_to_word):
    ingredients = [int(re.sub("ingr_", "", i)) for i in ingredients.split(";")]
    ingredients = [id_to_word[i] for i in ingredients]
    return " ".join(ingredients)


def preprocess(conf, df_type="train"):
    df_dish = pd.read_csv(conf.DF_DISH_PATH)
    df_ingr = pd.read_csv(conf.DF_INGREDIENTS_PATH)
    
    id_to_word = dict(zip(df_ingr["id"], df_ingr["ingr"]))
    df_dish["ingredients"] = df_dish["ingredients"].apply(lambda x: move_ingredients(x, id_to_word))

    df_dish["ingredients"] = (
    df_dish["ingredients"]
    .str.replace(r"\bdeprecated\b", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    )
    
    df_dish = df_dish[df_dish["ingredients"].ne("")].reset_index(drop=True)

    df_train = df_dish[df_dish["split"] == "train"].drop(columns="split").reset_index(drop=True)
    
    train, val = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=conf.SEED)

    scaler = StandardScaler()

    train["total_mass"] = scaler.fit_transform(train[["total_mass"]]).ravel()
    val["total_mass"] = scaler.transform(val[["total_mass"]]).ravel()

    if df_type == "train":
        return train, val
        
    else:
        test = df_dish[df_dish["split"] == "test"].drop(columns="split").reset_index(drop=True)
        test["total_mass"] = scaler.transform(test[["total_mass"]]).ravel()
        return test


def get_transforms(config, df_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if df_type == "train":
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            A.RandomCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-90, 90), scale=(0.7, 1.3), translate_percent=(-0.05, 0.05), fill=0, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0)
        ], seed=config.SEED)

    else:
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0)
        ], seed=config.SEED)

    return transforms


def prepare_dataloader(conf, df, tokenizer, df_type="train"):
    if df_type == "train":
        train_transforms = get_transforms(conf, df_type="train")
        ds = MultimodalDataset(df, conf, train_transforms)
        dl = DataLoader(ds, batch_size=conf.BATCH_SIZE, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer))

    else:
        val_transforms = get_transforms(conf, df_type="val")
        ds = MultimodalDataset(df, conf, val_transforms)
        dl = DataLoader(ds, batch_size=conf.BATCH_SIZE, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer))

    return dl
    