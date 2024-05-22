import pickle
import torch
from torchtext.data.metrics import bleu_score

def save_pkl(data, fname):
    # Save data to a pickle file
    with open(fname, "wb") as f:
        pickle.dump(data, f)

def load_pkl(fname):
    # Load data from a pickle file
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

def get_bleu_score(output, gt, vocab, specials, max_n=4):
    # Convert tensor to string tokens, excluding special tokens
    def itos(x):
        x = list(x.cpu().numpy())
        tokens = vocab.lookup_tokens(x)
        tokens = list(filter(lambda x: x not in {"", " ", "."} and x not in list(specials.keys()), tokens))
        return tokens

    # Get predictions and ground truth as string tokens
    pred = [out.max(dim=1)[1] for out in output]
    pred_str = list(map(itos, pred))
    gt_str = list(map(lambda x: [itos(x)], gt))

    # Calculate BLEU score
    score = bleu_score(pred_str, gt_str, max_n=max_n) * 100
    return score

def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    # Move source to device and create source mask
    src = src.to(model.device)
    src_mask = model.make_src_mask(src).to(model.device)
    memory = model.encode(src, src_mask)

    # Initialize target sequence with the start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    for i in range(max_len - 1):
        memory = memory.to(model.device)
        tgt_mask = model.make_tgt_mask(ys).to(model.device)
        src_tgt_mask = model.make_src_tgt_mask(src, ys).to(model.device)
        out = model.decode(ys, memory, tgt_mask, src_tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Append next word to target sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys
