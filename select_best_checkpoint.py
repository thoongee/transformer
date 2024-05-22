import os
import torch
from torch import nn
from models.build_model import build_model
from data import Multi30k
from utils import get_bleu_score

DATASET = Multi30k()

def evaluate(model, data_loader, criterion):
    model.eval()  # Set model to evaluation mode
    epoch_loss = 0
    total_bleu = []
    with torch.no_grad():  # Disable gradient calculation
        for idx, (src, tgt) in enumerate(data_loader):
            src = src.to(model.device)  # Move source to device (GPU/CPU)
            tgt = tgt.to(model.device)  # Move target to device (GPU/CPU)
            tgt_x = tgt[:, :-1]  # Decoder input (all tokens except last)
            tgt_y = tgt[:, 1:]   # Decoder target (all tokens except first)

            output, _ = model(src, tgt_x)  # Forward pass

            # Reshape for loss computation
            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)  # Compute loss

            epoch_loss += loss.item()  # Accumulate loss
            # Calculate BLEU score for current batch
            score = get_bleu_score(output, tgt_y, DATASET.vocab_tgt, DATASET.specials)
            total_bleu.append(score)
        num_samples = idx + 1  # Number of samples processed

    loss_avr = epoch_loss / num_samples  # Average loss
    bleu_score = sum(total_bleu) / len(total_bleu)  # Average BLEU score
    return loss_avr, bleu_score  # Return average loss and BLEU score

def main(checkpoint_dir, best_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=device, dr_rate=0.1)  # Build model
    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)  # Loss function

    # Data loaders for training, validation, and test sets
    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=512, num_workers=2)
    
    best_loss = float('inf')  # Initialize best loss
    best_bleu = 0  # Initialize best BLEU score
    best_checkpoint = None  # Initialize best checkpoint

    # Iterate over checkpoint files
    for file_name in os.listdir(checkpoint_dir):
        if file_name.endswith('.pt'):
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            checkpoint = torch.load(checkpoint_path)  # Load checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])  # Load model state

            valid_loss, bleu_score = evaluate(model, valid_iter, criterion)  # Evaluate model
            print(f"Checkpoint {file_name} - Valid Loss: {valid_loss:.5f}, BLEU Score: {bleu_score:.5f}")

            # Update best checkpoint if current is better
            if valid_loss < best_loss or (valid_loss == best_loss and bleu_score > best_bleu):
                best_loss = valid_loss
                best_bleu = bleu_score
                best_checkpoint = checkpoint_path

    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint} with loss {best_loss:.5f} and BLEU score {best_bleu:.5f}")
        torch.save(torch.load(best_checkpoint), best_model_path)  # Save best model

if __name__ == "__main__":
    import argparse

    # Argument parser for command line inputs
    parser = argparse.ArgumentParser(description="Select the best checkpoint based on validation loss and BLEU score.")
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing checkpoint files.')
    parser.add_argument('--best-model-path', type=str, required=True, help='Path to save the best model.')

    args = parser.parse_args()  # Parse arguments
    main(args.checkpoint_dir, args.best_model_path)  # Run main function with parsed arguments

