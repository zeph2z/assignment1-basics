from my_answer.train import TransformerLM, model_hp, train_hp, vocab_path, merges_path, ckpt_path, Tokenizer
import torch, os
from my_answer.SDPA import softmax

tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path)
prompt = None

def get_prompt():
    input_str = input("请输入文本：")
    prompt = tokenizer.encode(input_str)
    return prompt

def generate(device=train_hp["device"],
        context_length=train_hp["context_length"],
        model_hp=model_hp, prompt=prompt, temp=0.8, p=0.9):
    # setup
    model = TransformerLM(**model_hp)
    ckpt_file = os.path.join(ckpt_path, f"ckpt_295.pt")
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # prompt
    prompt = torch.Tensor(prompt).unsqueeze(0).to(device=device, dtype=torch.long)

    # generate
    output = prompt
    context = prompt
    
    for i in range(100):
        with torch.no_grad():
            logits = model.forward(context) # [batch_size, context_length, vocab_size]
        logits = logits[0, -1, :] # [vocab_size, ]

        if temp == 0:
            next_id = torch.argmax(logits, keepdim=True)
        else:
            logits /= temp
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cum = softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = (cum - softmax(sorted_logits, dim=-1)) > p
            sorted_logits[remove] = -torch.inf
            logits = torch.scatter(logits, 0, sorted_idx, sorted_logits)

            # sample
            probs = softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

        output = torch.cat([output, next_id.unsqueeze(0)], dim=1)
        if output.shape[1] > context_length:
            context = output[:, -context_length:]

    output = tokenizer.decode(output[0].tolist())
    print(output)

if __name__ == "__main__":
    prompt = get_prompt()
    generate(prompt=prompt)