# data/data_loader.py
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .dataset import EPRNSimpleDataset
from config import config

def create_data_loaders(dataset, batch_size=None):
    """Create train, validation, and test data loaders"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Split data
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=1-config.TRAIN_RATIO, random_state=config.SEED,
        stratify=dataset.subject_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=config.TEST_RATIO/(config.VAL_RATIO+config.TEST_RATIO),
        random_state=config.SEED, stratify=dataset.subject_labels[temp_idx]
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)