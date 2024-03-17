from typing import List
from pathlib import Path
import json
import numpy as np
import random
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from sentencepiece import SentencePieceProcessor

## https://pytorch.org/xla/release/2.1/index.html#pytorch-on-xla-devices
# import torch_xla.core.xla_model as xm

class bcolors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'


llama2_path = '../llama'
sys.path.append(llama2_path)
import llama
from llama.model import ModelArgs, Transformer
from llama import Llama, Dialog

def sample_top_p(probs, p):
  """
  Perform top-p (nucleus) sampling on a probability distribution.

  Args:
      probs (torch.Tensor): Probability distribution tensor.
      p (float): Probability threshold for top-p sampling.

  Returns:
      torch.Tensor: Sampled token indices.

  Note:
      Top-p sampling selects the smallest set of tokens whose cumulative probability mass
      exceeds the threshold p. The distribution is renormalized based on the selected tokens.

  """
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  mask = probs_sum - probs_sort > p
  probs_sort[mask] = 0.0
  probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
  next_token = torch.multinomial(probs_sort, num_samples=1)
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    traindata_path: str = "the_prince.txt"
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """

    sp_model = SentencePieceProcessor(model_file=tokenizer_path)

    with open(traindata_path, 'r') as f:
        data = f.read()

    all_tokens = []
    text = data.strip()
    tokens = sp_model.encode(text)
    tokens = [sp_model.bos_id()] + tokens
    all_tokens.extend(tokens)

    # convert to uint16 np array
    import numpy as np
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    with open('the_prince.bin', 'wb') as f:
        f.write(all_tokens.tobytes())

    print("[START] training tiny llama2 on Apple M1 processor")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
      params = json.loads(f.read())

    model_config = ModelArgs(**params)
    model = Transformer(model_config)

    numParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("numParameters = ", numParameters)

    # initialize model weights
    model.apply(lambda m : nn.init.xavier_uniform_(m.weight.data) if hasattr(m, 'weight') and m.weight.dim() > 1 else None)

    # Load training samples
    BS = model_config.max_batch_size
    train_fn = "the_prince.bin"
    m = np.memmap(train_fn, dtype=np.uint16, mode='r')
    max_seq_len = model_config.max_seq_len
    num_batches = len(m) // max_seq_len
    num_batches -= BS
    assert num_batches > 0

    ixs = list(range(num_batches))
    random.Random(42).shuffle(ixs)
    train_data = []
    for ix in ixs:
        start = ix * max_seq_len
        end = start + max_seq_len * BS + 1
        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
        x, y = chunk[:-1], chunk[1:]
        x, y = x.reshape(BS, -1), y.reshape(BS, -1)
        train_data.append((x, y))

    print("Train data size = ", len(train_data))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    lossFn = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0
    count = 0
    n_epochs = 5
    for epoch in range(n_epochs):
        for X, Y in train_data:
          optimizer.zero_grad()
          # X = X.to(device)
          # Y = Y.to(device)
          logits = model(X, start_pos=0)
          targets = Y
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          total_loss += loss.item()
          # xm.mark_step()
          if count % 367 == 0:
            print("data point: ", count, ", loss = ", loss.item())
            avg_loss = total_loss / len(train_data)
            # checkpointing model
            print("epoch ", epoch, ": loss = ", avg_loss)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        }, "model_ckpt.pt")
          count += 1

if __name__ == "__main__":
  ckpt_dir = "llama2-tiny"
  # tokenizer_path = "tokenizer.model"
  tokenizer_path = "codellama7b_tokenizer.model"
  device = "cpu"

  do_inference: bool = True 
  do_training: bool = not do_inference

  if do_training:
    # Task 0: Train llama2 to read <<the prince>>.
    # main(ckpt_dir="llama-2-tiny", tokenizer_path=tokenizer_path)

    # Task 1: Train llama2 to code like llvm
    main(ckpt_dir="llama2-tiny", tokenizer_path=tokenizer_path, traindata_path="samplecode.txt")

  elif do_inference:
    # top-p sampling temperature:
    temperature = 0.6
    top_p = 0.9

    # Load from model checkpoint and see what llama has learnt.
    # TODO: implement model beam search sampling
    with open(Path(ckpt_dir) / "params.json", "r") as f:
      params = json.loads(f.read())

    checkpoint = torch.load("model_ckpt.pt")
    model_config = ModelArgs(**params)
    model = Transformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    sp_model = SentencePieceProcessor(model_file=tokenizer_path)

    max_seq_len = model_config.max_seq_len
    prompt_text= "import math\ndef gridsearch("
    raw_tokens = sp_model.encode(prompt_text)
    #seed_bpe = [sp_model.decode(token) for token in raw_tokens]
    #print(prompt_text)
    #print("========= after tokenization =========")
    #print(seed_bpe)
    max_gen_len = 2 * model_config.max_seq_len
    prompt_tokens = []
    prompt_tokens.append(raw_tokens)
    bsz0 = len(prompt_tokens)
    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= model_config.max_seq_len
    total_len = min(model_config.max_seq_len, max_gen_len + max_prompt_len)
    pad_id = sp_model.pad_id()
    tokens = torch.full((bsz0, total_len), pad_id, dtype=torch.long, device=device)
    for k,t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz0, device=device)
    input_text_mask = tokens != pad_id
    for cur_pos in range(min_prompt_len, total_len):
      logits = model(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
      if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
      else:
        next_token = torch.argmax(logits[:, -1], dim=-1)
      next_token = next_token.reshape(-1)
      # only replace token if prompt has already been generated
      next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
      tokens[:, cur_pos] = next_token
      prev_pos = cur_pos

      eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == sp_model.eos_id())
      if all(eos_reached):
        break

    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
      # cut to max gen len
      echo = False
      start = 0 if echo else len(prompt_tokens[i])
      toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
      if sp_model.eos_id() in toks:
        eos_idx = toks.index(sp_model.eos_id())
        toks = toks[:eos_idx]
      out_tokens.append(toks)

    sampled_text = [sp_model.decode(t) for t in out_tokens]
    # Generated text from sampling is underlined.
    print(prompt_text + bcolors.UNDERLINE + sampled_text[0] + bcolors.ENDC)

  else:
    # Shouldn't reach here.
    assert False, "Set do_training or do_inference"

