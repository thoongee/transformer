import os, sys, time
import logging

import torch
from torch import nn, optim

from config import *
from models.build_model import build_model
from data import Multi30k
from utils import get_bleu_score, greedy_decode

torch.cuda.empty_cache() # Free up gpu memory
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' #  reduce memory fragmentation

DATASET = Multi30k()


def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0

    for idx, (src, tgt) in enumerate(data_loader):
        # Move data to the model's device (e.g., GPU)
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        tgt_x = tgt[:, :-1]  # Input to the decoder
        tgt_y = tgt[:, 1:]   # Target output

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output, _ = model(src, tgt_x)

        # Flatten the outputs and targets for loss computation
        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)  # Compute the loss
        loss.backward()  # Backpropagation
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()  # Update model parameters

        epoch_loss += loss.item()  # Accumulate the loss

    num_samples = idx + 1

    # Save checkpoint if directory is provided
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{epoch:04d}.pt")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                   }, checkpoint_file)

    return epoch_loss / num_samples  # Return average loss per sample


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    total_bleu = []

    with torch.no_grad():
        for idx, (src, tgt) in enumerate(data_loader):
            # Move data to the model's device (e.g., GPU)
            src = src.to(model.device)
            tgt = tgt.to(model.device)
            tgt_x = tgt[:, :-1]  # Input to the decoder
            tgt_y = tgt[:, 1:]   # Target output

            # Forward pass
            output, _ = model(src, tgt_x)

            # Flatten the outputs and targets for loss computation
            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)  # Compute the loss

            epoch_loss += loss.item()  # Accumulate the loss
            score = get_bleu_score(output, tgt_y, DATASET.vocab_tgt, DATASET.specials)  # Calculate BLEU score
            total_bleu.append(score)

        num_samples = idx + 1

    # Calculate average loss and BLEU score
    loss_avr = epoch_loss / num_samples
    bleu_score = sum(total_bleu) / len(total_bleu)
    return loss_avr, bleu_score  # Return average loss and BLEU score


def main():
    # Build the model
    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)

    def initialize_weights(model):
        # Initialize model weights
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)

    model.apply(initialize_weights)  # Apply weight initialization

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)  # Define loss function

    # Get data iterators for training, validation, and testing
    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    for epoch in range(N_EPOCH):
        logging.info(f"*****epoch: {epoch:02}*****")
        train_loss = train(model, train_iter, optimizer, criterion, epoch, CHECKPOINT_DIR)
        logging.info(f"train_loss: {train_loss:.5f}")
        valid_loss, bleu_score = evaluate(model, valid_iter, criterion)
        if epoch > WARM_UP_STEP:
            scheduler.step(valid_loss)  # Update learning rate
        logging.info(f"valid_loss: {valid_loss:.5f}, bleu_score: {bleu_score:.5f}")

        # Log translation example
        logging.info(DATASET.translate(model, "A little girl climbing into a wooden playhouse .", greedy_decode))
        # Expected output: "Ein kleines Mädchen klettert in ein Spielhaus aus Holz ."

    # Evaluate the model on the test set
    test_loss, bleu_score = evaluate(model, test_iter, criterion)
    logging.info(f"test_loss: {test_loss:.5f}, bleu_score: {bleu_score:.5f}")

if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    main()
