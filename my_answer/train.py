import numpy, os
import torch.nn as nn
from my_answer.tokenizer import Tokenizer
from my_answer.rmsnorm import RMSNorm
from my_answer.MHA import MHA
from my_answer.swiglu import SwiGLU
from my_answer.rope import RoPE
from my_answer.embedding import Embedding
from my_answer.linear import Linear
from my_answer.AdamW import AdamW
from tqdm import tqdm
import tests.adapters as ad

train_path = "data/TinyStoriesV2-GPT4-train.txt"
valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
vocab_path = "data/TinyStoriesVocab.json"
merges_path = "data/TinyStoriesMerges.json"
input_path = "data/TinyStoriesEncodeArray.bin.npy"
ckpt_path = "data/checkpoints"

model_hp = dict(
    vocab_size=10000,
    context_length=256,
    d_model=64,
    num_heads=8,
    d_ff=192,
    num_layers=4,
    theta=10000,
    device="mps",
    eps=1e-5
)

train_hp = dict(
    context_length=256,
    batch_size=16,
    total_iters=301,
    max_lr=1e-3,
    min_lr=1e-4,
    warmup_iters=20,
    betas=(0.9, 0.999),
    eps=1e-8,          # AdamW 的 eps，属于训练
    weight_decay=0.1,
    grad_clip=1.0,
    device="mps",
    cosine_cycle_iters=250,
    save_iters = 50
)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, eps, device, num_heads, d_ff, theta, context_length):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, eps=eps, device=device)
        self.ln2 = RMSNorm(d_model=d_model, eps=eps, device=device)
        self.mha = MHA(d_in=d_model, d_out=d_model, num_head=num_heads, device=device)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device)
        self.rope = RoPE(theta=theta, d_k = d_model // num_heads, max_seq_len=context_length, device=device)
        
    def forward(self, x):
        residual = x
        x = self.ln1.forward(x)
        x = self.mha.forward(x, rope=self.rope)
        x = residual + x

        residual = x
        x = self.ln2.forward(x)
        x = self.ffn.forward(x)
        x = residual + x
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, eps, device, num_heads, d_ff, theta, context_length, num_layers):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, eps, device, num_heads, d_ff, theta, context_length) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps, device)
        self.lm_head = Linear(d_model, vocab_size, device)

    def forward(self, x):
        x = self.embedding.forward(x) # [batch_size, context_length, d_model]
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i].forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x) # [batch_size, context_length, vocab_size]
        return x

def train(batch_size, total_iters, max_lr, min_lr, warmup_iters, betas, eps, weight_decay, grad_clip, device, model_hp, context_length, cosine_cycle_iters, save_iters):
    # setup
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path)
    assert len(tokenizer.vocab) == model_hp["vocab_size"]

    dataset = numpy.load(input_path)
    model = TransformerLM(**model_hp)
    opt = AdamW(model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay)

    for it in tqdm(range(total_iters), desc="training"):
        # [batch_size, context_length]
        x, y = ad.run_get_batch(dataset, batch_size, context_length, device)
        logits = model.forward(x) # [batch_size, context_length, vocab_size]

        opt.zero_grad(set_to_none=True)
        loss = ad.run_cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        loss.backward()
        ad.run_gradient_clipping(model.parameters(), max_l2_norm=grad_clip)
        lr = ad.run_get_lr_cosine_schedule(it=it, max_learning_rate=max_lr, min_learning_rate=min_lr, warmup_iters=warmup_iters, cosine_cycle_iters=cosine_cycle_iters)
        opt.param_groups[0]["lr"] = lr
        opt.step()

        if it % save_iters == 0:
            print(f"iteration {it}: loss = {loss.item()}")
            os.makedirs(ckpt_path, exist_ok=True)
            ckpt_file = f"{ckpt_path}/ckpt_{it}.pt"
            ad.run_save_checkpoint(model, opt, it, ckpt_file)

if __name__ == "__main__":
    train(**train_hp, model_hp=model_hp)