import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class AnimalDataConfig:
    N_CLASSES = 10
    IMG_SIZE = 224
    ID2LABEL = {
        0: 'butterfly',
        1: 'cat',
        2: 'chicken',
        3: 'cow',
        4: 'dog',
        5: 'elephant',
        6: 'horse',
        7: 'sheep',
        8: 'spider',
        9: 'squirrel'
    }
    LABEL2ID = {
        'butterfly': 0,
        'cat': 1,
        'chicken': 2,
        'cow': 3,
        'dog': 4,
        'elephant': 5,
        'horse': 6,
        'sheep': 7,
        'spider': 8,
        'squirrel': 9
    }

class ModelConfig:
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_NAME = 'animal'
    MODEL_WEIGHT = ROOT_DIR / 'models' / 'weights' / 'best.pt'
    DEVICE = 'cpu'