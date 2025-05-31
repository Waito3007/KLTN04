import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
from ai.models.hierarchical_attention import HierarchicalAttentionNetwork
from ai.data_preprocessing.text_processor import TextProcessor
from ai.data_preprocessing.embedding_loader import EmbeddingLoader
from ai.training.multitask_trainer import MultiTaskTrainer
from ai.evaluation.metrics_calculator import calc_metrics
from ai.train_han_multitask import CommitDataset, load_commit_data  # Add this import
import numpy as np
import json
from datetime import datetime
import random

def continue_training(existing_model_path, new_data_path, device=None, num_epochs=5):
    """Tiếp tục training model với dữ liệu mới"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading existing model from {existing_model_path}")
    checkpoint = torch.load(existing_model_path, map_location=device)
    
    # Load the new data
    print(f"Loading new data from {new_data_path}")
    new_samples, new_mappings = load_commit_data(new_data_path)
    
    # Merge mappings from checkpoint and new data
    merged_author_map = checkpoint['author_map'].copy()
    merged_repo_map = checkpoint['repo_map'].copy()

    # Add new authors and repos
    max_author_idx = max(merged_author_map.values(), default=-1)
    max_repo_idx = max(merged_repo_map.values(), default=-1)

    for author in new_mappings['author_map']:
        if author not in merged_author_map:
            max_author_idx += 1
            merged_author_map[author] = max_author_idx

    for repo in new_mappings['repo_map']:
        if repo not in merged_repo_map:
            max_repo_idx += 1
            merged_repo_map[repo] = max_repo_idx

    # Initialize data processing
    processor = TextProcessor()
    embed_loader = EmbeddingLoader(embedding_type='codebert')
    embed_loader.load()
    
    # Create datasets with merged mappings
    random.shuffle(new_samples)
    train_size = int(0.8 * len(new_samples))
    train_samples = new_samples[:train_size]
    val_samples = new_samples[train_size:]

    print("Creating datasets...")
    train_dataset = CommitDataset(
        train_samples,
        processor,
        embed_loader,
        author_map=merged_author_map,
        repo_map=merged_repo_map
    )

    # Update num_classes_dict with the number of classes for authors
    num_classes_dict = {
        'purpose': len(train_dataset.purpose_map),
        'tech_tag': len(train_dataset.tech_vocab),
        'suspicious': 2,
        'sentiment': len(train_dataset.sentiment_map),
        'author': len(merged_author_map),
        'source_repo': len(merged_repo_map),
        'commit_type': len(train_dataset.commit_type_map)
    }

    # Initialize model with updated number of classes
    print("Initializing model...")
    if embed_loader.embedding_type == 'codebert':
        embed_dim = embed_loader.model.config.hidden_size
    else:
        embed_dim = 768

    model = HierarchicalAttentionNetwork(
        embed_dim=embed_dim,
        hidden_dim=128,
        num_classes_dict=num_classes_dict
    ).to(device)

    # Load existing weights where they match
    existing_state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Copy matching weights from existing model
    for name, param in existing_state_dict.items():
        if name in model_dict:
            # Nếu shape khớp, copy weights
            if model_dict[name].shape == param.shape:
                model_dict[name].copy_(param)
            else:
                print(f"Warning: Shape mismatch for {name}, initializing randomly")
    
    model.load_state_dict(model_dict)

    # Enable CUDA optimizations if using GPU
    if device.type == 'cuda':
        print("Enabling CUDA optimizations...")
        torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

    # Setup training parameters
    batch_size = 64 if device.type == 'cuda' else 8
    num_workers = 6 if device.type == 'cuda' else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = None
    if val_samples:
        print("Creating validation dataset...")
        val_dataset = CommitDataset(
            val_samples,
            processor,
            embed_loader,
            author_map=merged_author_map,
            repo_map=merged_repo_map
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        print("No validation data provided.")

    # Setup optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Giảm learning rate cho fine-tuning
    
    loss_fns = {
        'purpose': torch.nn.CrossEntropyLoss(),
        'suspicious': torch.nn.CrossEntropyLoss(),
        'tech_tag': torch.nn.CrossEntropyLoss(),
        'sentiment': torch.nn.CrossEntropyLoss(),
        'author': torch.nn.CrossEntropyLoss(),
        'source_repo': torch.nn.CrossEntropyLoss(),
        'commit_type': torch.nn.CrossEntropyLoss()
    }

    # Initialize trainer
    print("Initializing trainer...")
    trainer = MultiTaskTrainer(model, optimizer, loss_fns, device=device)

    # Setup logging
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"incremental_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Incremental Training Configuration\n{'='*30}\n")
        f.write(f"Base model: {existing_model_path}\n")
        f.write(f"New data: {new_data_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of workers: {num_workers}\n")
        f.write(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_samples else 0}\n")
        f.write(f"Model classes: {num_classes_dict}\n\n")

    # Training loop
    best_val_loss = float('inf')
    current_val_loss = float('inf')

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = trainer.train_epoch(train_loader)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

            # Validation
            task_metrics = {}
            if val_loader:
                current_val_loss, preds, true_labels = trainer.validate(val_loader)
                print(f"Epoch {epoch+1}: Val Loss = {current_val_loss:.4f}")

                for task in preds:
                    if len(np.unique(true_labels[task].numpy())) > 1:
                        metrics = calc_metrics(true_labels[task].numpy(), preds[task].argmax(-1).numpy())
                    else:
                        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0}
                    task_metrics[task] = metrics

            # Save best model
            if val_loader and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_path = os.path.join(base_dir, "models", "han_multitask_incremental_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'author_map': merged_author_map,
                    'repo_map': merged_repo_map,
                    'num_classes_dict': num_classes_dict
                }, save_path)
                print(f"Epoch {epoch+1}: New best model saved with val_loss: {best_val_loss:.4f}")

            if device.type == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        save_path = os.path.join(base_dir, "models", "han_multitask_incremental_final.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'author_map': merged_author_map,
            'repo_map': merged_repo_map,
            'final_val_loss': current_val_loss,
            'num_classes_dict': num_classes_dict
        }, save_path)
        print(f"\nFinal model saved to {save_path}")

def main():
    # Parse command line arguments for model path and new data path
    if len(sys.argv) != 3:
        print("Usage: python train_han_multitask_incremental.py <existing_model_path> <new_data_path>")
        sys.exit(1)

    existing_model_path = sys.argv[1]
    new_data_path = sys.argv[2]

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Start incremental training
    continue_training(existing_model_path, new_data_path, device)

if __name__ == "__main__":
    main()
