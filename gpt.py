import argparse
import sentencepiece as spm
import sys
import torch
import torch.nn as nn
from contextlib import redirect_stdout
from torch.nn import functional as F
import requests
from warcio.archiveiterator import ArchiveIterator
import io
import gzip

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2
save_path = 'gpt_language_model.pth'
# ------------

# Encoding function
def encode(text):
    return sp.encode(text, out_type=int)

# Decoding function
def decode(tokens):
    return sp.decode(tokens)

def fetch_file_from_https(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return io.BytesIO(response.content)

def extract_text_from_warc(warc_buffer):
    text = []
    for record in ArchiveIterator(warc_buffer):
        if record.http_headers and record.http_headers.get_header('Content-Type') == 'text/html':
            payload = record.content_stream().read()
            html = payload.decode('utf-8', errors='ignore')
            text.append(html)
    return text

def stream_data_from_warc_files(warc_paths, max_size_gb=2):
    print("Stream data...")
    i = 0
    total_size = 0
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    for warc_path in warc_paths:
        i = i + 1
        print(f"Processing path {i}")
        warc_url = f'https://data.commoncrawl.org/{warc_path}'
        response = requests.get(warc_url, stream=True)
        response.raise_for_status()
        with io.BytesIO(response.content) as warc_buffer:
            for chunk in extract_text_from_warc(warc_buffer):
                chunk_size = len(chunk)
                total_size += chunk_size
                print(f"Total size: {total_size} bytes")
                if (total_size > max_size_bytes):
                    return
                yield chunk

def get_warc_paths():
    # Download the WARC paths list
    paths_url = 'https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-33/warc.paths.gz'
    response = requests.get(paths_url, stream=True)
    response.raise_for_status()
    with io.BytesIO(response.content) as f:
        with gzip.open(f, 'rt') as g:
            return [line.strip() for line in g]

# Create training and validation data
def create_data_stream():
    warc_paths = get_warc_paths()
    encoded_text = []
    i = 0
    for chunk in stream_data_from_warc_files(warc_paths):
        encoded_chunk = encode(chunk)
        encoded_text.extend(chunk)
    data = torch.tensor(encoded_text, dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% for training, rest for validation
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
    input='input.txt',
    model_prefix='spm_model',
    vocab_size=10770
    )
    print("Finish creating data stream")
    return data[:n], data[n:]

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file='spm_model.model')

train_data, val_data = create_data_stream()

vocab_size = sp.get_piece_size()

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def train_model(self):
        print("Beginning GPT training...")
        @torch.no_grad()
        def estimate_loss(model):
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss(self)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            X, Y = get_batch('train')
            logits, loss = self(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % eval_interval == 0:
                torch.save(self.state_dict(), save_path)

if __name__ == "__main__":
    model = GPTLanguageModel().to(device)
    model.train_model()
