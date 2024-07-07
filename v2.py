import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8 # the maximim context length for prediction
max_iters = 3000
eval_iters = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
# ----

torch.manual_seed(1337)

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda x: [stoi[ch] for ch in x] # encoder: take a string, output a list of integers
decode = lambda x: ''.join([itos[i] for i in x]) # decoder: take a list of integers, output a string

# train and test splits

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

#data loading
def get_batch(split):
    # generate a small batch of data of input x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # when load the model to the device, the data should be on the same device
    x = x.to(device)
    y = y.to(device)
    return x, y

# no_grad means that the operations inside the block will not be recorded for backpropagation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets = None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx) # B, T, n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, n_embed
        x = tok_emb + pos_emb # B, T, n_embed
        logits = self.lm_head(x) # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            # idx and targets are both (B,T) tensor of integers
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # B, C
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # B, C
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # B, 1
            # append sampled index to the running sequence
            idx = torch.cat([idx, next_token], dim=1)
        return idx
        

model = BigramLanguageModel().to(device)
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'iter {iter} | train loss {losses["train"]} | val loss {losses["val"]}')
    
    #sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros(1, 1, dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
