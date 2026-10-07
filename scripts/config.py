

class Config:
    # seed
    SEED = 42
    
    # Названия моделей
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # Слои на разморозку
    TEXT_MODEL_UNFREEZE = "encoder.layer.10|encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"

    # Параметры обучения
    BATCH_SIZE = 4
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    NUM_LR = 1e-3
    REGRESSOR_LR = 5e-4
    EPOCHS = 30
    PATIENCE = 3 # кол-во эпох без улучшений для ранней остановки
    DROPOUT = 0.15
    HIDDEN_DIM = 256
    NUM_DIM = 32

    # Пути 
    SAVE_PATH = "best_model.pth"
    DF_DISH_PATH = "data/dish.csv"
    DF_INGREDIENTS_PATH = "data/ingredients.csv"
    IMAGE_ROOT = "data/images"
    IMAGE_NAME = "rgb.png"

conf = Config()