from typing import List
from pathlib import Path
import json
import numpy as np
import random
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from sentencepiece import SentencePieceProcessor
import llama
from llama.model import ModelArgs, Transformer
from llama import Llama, Dialog

## https://pytorch.org/xla/release/2.1/index.html#pytorch-on-xla-devices
# import torch_xla.core.xla_model as xm

# Define underline style used for inference output.
class bcolors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'

# Define utility function for top-p sampling
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

# Define the main function for pre-training a Llama2 model
def main(
    param_dir: str,
    tokenizer_path: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    n_epochs: int = 10,
    traindata_path: str = "",
    ckpt_path: str = "model_ckpt.pt",
):
    """
    Pre-train a Llama2 model using local training data.

    Args:
        param_dir (str): The directory containing model configuration json file.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.3.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
        n_epochs (int, optional): The number of epochs to train the model. Defaults to 10.
        traindata_path (str, optional): The path to the training data file. 
        ckpt_path (str, optional): The path to save the model checkpoint. Defaults to "model_ckpt.pt".
    """
    # Use SentencePiece to encode/decode the text data
    sp_model = SentencePieceProcessor(model_file=tokenizer_path)
    # Load the train data
    if traindata_path == "":
       print("No training data path provided. Exiting.")
       return
  
    with open(traindata_path, 'r') as f:
      data = f.read()

    all_tokens = []
    text = data.strip()
    tokens = sp_model.encode(text)
    tokens = [sp_model.bos_id()] + tokens
    all_tokens.extend(tokens)

    # convert to uint16 np array to save space when saving to disk
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    with open('traindata.bin', 'wb') as f:
        f.write(all_tokens.tobytes())

    print("[START] Training tiny Llama2 model on Apple M1 processor:")
    # Load the Llama2 Transformer model
    with open(Path(param_dir) / "params.json", "r") as f:
      params = json.loads(f.read())

    model_config = ModelArgs(**params)
    model = Transformer(model_config)

    numParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("numParameters = ", numParameters)

    # Initialize model weights
    model.apply(lambda m : nn.init.xavier_uniform_(m.weight.data) if hasattr(m, 'weight') and m.weight.dim() > 1 else None)

    # Load the training samples
    BS = model_config.max_batch_size
    train_fn = "traindata.bin"
    m = np.memmap(train_fn, dtype=np.uint16, mode='r')
    max_seq_len = model_config.max_seq_len
    num_batches = len(m) // max_seq_len
    num_batches -= BS
    assert num_batches > 0

    ixs = list(range(num_batches))
    random_seed = 42
    random.Random(random_seed).shuffle(ixs)
    train_data = []
    for ix in ixs:
        start = ix * max_seq_len
        end = start + max_seq_len * BS + 1
        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
        x, y = chunk[:-1], chunk[1:]
        x, y = x.reshape(BS, -1), y.reshape(BS, -1)
        train_data.append((x, y))

    print("Train data size = ", len(train_data))
    # Start model training
    model.train()
    learning_rt = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rt)
    lossFn = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0
    count = 0
    
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
          # Print data point level results to show progress. Not needed when the code has been tested and works properly.
          #if count % 10 == 0:
          #  print("data point: ", count, ", loss = ", loss.item())
          
          if count > 0:
            avg_loss = total_loss / count
          else:
            avg_loss = total_loss 
            
          #  print("epoch ", epoch, ": average loss = ", avg_loss)
            # checkpointing model
          torch.save({'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': avg_loss,
                      }, ckpt_path)
          count += 1
        #avg_loss = total_loss / len(train_data)
        print("epoch ", epoch, ": average loss = ", avg_loss)


def do_inference(
    param_dir: str,
    tokenizer_path: str,
    ckpt_path: str,
    prompt_text: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    device: str = "cpu"
):
    """
    Making inference from a prompt using pre-trained Llama2 model.

    Args:
        param_dir (str): The directory containing the model configration json file.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        ckpt_path (str): The directory containing checkpoint files for the pretrained model.
        prompt_text (str): The prompt text for generating the next token.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.3.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        device (str, optional): The device to run inference on. Defaults to "cpu".  
    """

    # Load from model checkpoint and see what Llama has learnt.
    # TODO: implement model beam search sampling
    with open(Path(param_dir) / "params.json", "r") as f :
      params = json.loads(f.read())

    checkpoint = torch.load(ckpt_path)
    model_config = ModelArgs(**params)
    model = Transformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    sp_model = SentencePieceProcessor(model_file=tokenizer_path)

    # Encode Prompt Text
    raw_tokens = sp_model.encode(prompt_text) # Encode the prompt text
    max_gen_len = 2 * model_config.max_seq_len
    prompt_tokens = []
    prompt_tokens.append(raw_tokens)
    bsz0 = len(prompt_tokens)
    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= model_config.max_seq_len
    total_len = min(model_config.max_seq_len, max_gen_len + max_prompt_len)
    pad_id = sp_model.pad_id() # Padding
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
    
    # Original prompt is underlined followed by generated text.
    print(bcolors.UNDERLINE + prompt_text + bcolors.ENDC + sampled_text[0])


# Set up the model parameters to be taken from command line arguments
parser = argparse.ArgumentParser(description='Tiny Llama2 Pre-training or Inferencing.')
# parser.add_argument('doinference', type = bool,const=None, default=None,\
#                     help = 'Toggle indicating doing inference (True) or pre-training (False)')  # do_inference (True) or do_training (False) flag
parser.add_argument('--inference_only', action='store_true', help ='Toggle for turning to inference only mode. If not used,\
                    the program will perform model pre-training')
parser.add_argument('--param_dir', type = str, default = 'llama2-tiny', help = 'path to model configuration json file')           
parser.add_argument('--tokenizer_path', type = str, default = 'tokenizer.model',\
                    help = 'path to the tokenizer model used for text encoding/decoding')
parser.add_argument('--device', type = str, default = 'cpu', help = 'Device type: cpu or gpu')           
parser.add_argument('--ckpt_path', type = str, default = 'model_ckpt.pt', help = 'full path to the model checkpoint file') 
parser.add_argument('--n_epochs', type = int, default = 10, help = 'Number of epochs for training')
parser.add_argument('--temperature', type = float, default = 0.3, help = 'Temperature for sampling')
parser.add_argument('--top_p', type = float, default = 0.9, help = 'Top-p sampling parameter')
parser.add_argument('--prompt_text', type = str, default = "How to manage model risk "\
                    , help = 'Prompt text for generating the next token')
parser.add_argument('--traindata', type = str, default = "sr1107a1.txt", help = 'name of the training data') 
args = parser.parse_args()

# Execute the program
if __name__ == "__main__":
  if args.inference_only:
    # Task: Generate code or text from prompt text
    do_inference(param_dir=args.param_dir,tokenizer_path=args.tokenizer_path, \
                 ckpt_path=args.ckpt_path, temperature = args.temperature, \
                top_p=args.top_p, prompt_text=args.prompt_text, device = args.device)
  else:
    # Task: Train llama2 to generate next token for text or Python code depending on the traindata and tokenizer used
    main(param_dir=args.param_dir, tokenizer_path=args.tokenizer_path, \
         top_p=args.top_p,temperature = args.temperature, n_epochs=args.n_epochs, \
          traindata_path=args.traindata)

