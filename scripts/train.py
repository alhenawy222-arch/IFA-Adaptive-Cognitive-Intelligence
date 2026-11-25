"""Minimal training loop. Runs on CPU or a free Colab GPU.

Usage:  python scripts/train.py --tokenizer char --steps 300
        python scripts/train.py --tokenizer gpt2 --data my_text.txt
"""
import argparse, os, sys, urllib.request
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifa_omega import IFAConfig, IFAOmega

TINY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path):
    """TODO(data): plug in your dataset here. Easiest swap: use TinyStories via
    `datasets.load_dataset("roneneldan/TinyStories", split="train[:2%]")` and
    join the 'text' column into one string. Default: tiny Shakespeare."""
    if path:
        return open(path, encoding="utf-8").read()
    if not os.path.exists("data.txt"):
        urllib.request.urlretrieve(TINY_URL, "data.txt")
    return open("data.txt", encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", choices=["char", "gpt2"], default="char")
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    a = ap.parse_args()

    text = load_text(a.data)
    if a.tokenizer == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        data, vocab, decode = torch.tensor(enc.encode(text)), enc.n_vocab, enc.decode
    else:
        chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text]); vocab = len(chars)
        decode = lambda ids: "".join(chars[i] for i in ids)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = IFAConfig(vocab_size=vocab, n_layers=a.layers, d_model=a.d_model,
                    max_seq_len=a.seq, n_heads=4)
    model = IFAOmega(cfg).to(dev)
    print(f"device={dev} params={model.num_params()/1e6:.2f}M vocab={vocab}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    n = int(0.9 * len(data)); train, val = data[:n], data[n:]

    def batch(d):
        ix = torch.randint(len(d) - a.seq - 1, (a.batch,))
        x = torch.stack([d[i:i + a.seq] for i in ix]); y = torch.stack([d[i + 1:i + a.seq + 1] for i in ix])
        return x.to(dev), y.to(dev)

    for step in range(a.steps + 1):
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                vl = model(*batch(val))[1].item()
            print(f"step {step} val_loss {vl:.3f}")
            model.train()
        _, loss = model(*batch(train))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, "ifa_omega.pt")
    out = model.generate(torch.zeros(1, 1, dtype=torch.long, device=dev), 200)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
