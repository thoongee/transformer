import os
import torch
from torch import nn
from models.build_model import build_model
from data import Multi30k
from utils import get_bleu_score

DATASET = Multi30k()

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    total_bleu = []
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(data_loader):
            src = src.to(model.device)
            tgt = tgt.to(model.device)
            tgt_x = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            output, _ = model(src, tgt_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, tgt_y, DATASET.vocab_tgt, DATASET.specials)
            total_bleu.append(score)
        num_samples = idx + 1

    loss_avr = epoch_loss / num_samples
    bleu_score = sum(total_bleu) / len(total_bleu)
    return loss_avr, bleu_score

def main(checkpoint_dir, best_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=device, dr_rate=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)

    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=512, num_workers=2)
    
    best_loss = float('inf')
    best_bleu = 0
    best_checkpoint = None

    for file_name in os.listdir(checkpoint_dir):
        if file_name.endswith('.pt'):
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            valid_loss, bleu_score = evaluate(model, valid_iter, criterion)
            print(f"Checkpoint {file_name} - Valid Loss: {valid_loss:.5f}, BLEU Score: {bleu_score:.5f}")

            if valid_loss < best_loss or (valid_loss == best_loss and bleu_score > best_bleu):
                best_loss = valid_loss
                best_bleu = bleu_score
                best_checkpoint = checkpoint_path

    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint} with loss {best_loss:.5f} and BLEU score {best_bleu:.5f}")
        torch.save(torch.load(best_checkpoint), best_model_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select the best checkpoint based on validation loss and BLEU score.")
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing checkpoint files.')
    parser.add_argument('--best-model-path', type=str, required=True, help='Path to save the best model.')

    args = parser.parse_args()
    main(args.checkpoint_dir, args.best_model_path)
