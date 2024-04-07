# Pre-train a Tiny Llama2 Model on Apple M1 Processor 

This repo requires a MacBook Pro equipped with the powerful Apple M-series processor.

A. Prerequisite:

You will need to download Llama2 locally from Meta. Here are the steps to follow:

1. Access a download request from Meta through this link: https://llama.meta.com/llama-downloads/
2. Select "Llama2 & Llama Chat" model if you'd like to train a model for text generation, or "Code Llama" model if you'd like to train a model for code generation. Please read through the terms and conditions and accept them.
3. You'll receive an email from "AI at Meta" with the subject line "Get Started with Llama 2" which contains detailed instructions about the download. There are multiple options for model weights, and you should select the smallest one if you have a regular MacBook. You can always work your way up later if you have a powerful machine, and I'd love to hear about how they compare.


B. To set up the environment:

1. Clone the repo to a local development directory using the terminal:
For example, under `~/dev`, run `git clone` like this:
```
~/dev$ git clone --recursive git@github.com:CindySun89/Pre-train_Llama2_on_AppleM1.git
```
2. Put the Llama2 model inside the development directory:
'~/dev/Pre-train_Llama2_on_AppleM1/'
'
3. Update Llama2's original model.py with my own adaption for training:

```
cd ~/dev/Pre-train_Llama2_on_AppleM1
cp model.py.for_training llama/llama/model.py
``` 

C. To pre-train the model: 
File next_token_prediction.py contains codes for both model pre-training and inference and the default execution is pre-training. The input parameters are served through command line arguments when you execute the command below:

1. Run the following command (assume you git clone the repo under `~/dev` as done in A.1:
```
PYTHONPATH=~/dev/Pre-train_Llama2_on_AppleM1/llama/ python3 next_token_prediction.py
```
    It will launch the model pre-training using the default configurations. If you'd like to customize it, consider adding the relevant arguments to the command above. For example, if you'd like to run 100 epochs and a temperature of 0.2, you can run this: 
```
PYTHONPATH=~/dev/Pre-train_Llama2_on_AppleM1/llama/ python3 next_token_prediction.py --n_epochs 100 --temperature 0.2
```
    The default training data is sr1107a1.txt which was downloaded from https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf and converted to a .txt file. You can use your own training data and try it out. Make sure you add `--traindata <filename>.txt' to the command above to reflect such customizations.

    
2. TODO:

Describe how to collect sample data.

D. To make inference using the pre-trained model: by default the model checkpoint file is saved as model_ckpt.pt in the same development directory.

1. Run the following cmd (assume you git clone the repo under `~/dev` as done in A.1):
```
PYTHONPATH=~/dev/Pre-train_Llama2_on_AppleM1/llama/ python3 next_token_prediction.py --inference_only
```
Like in C.1, you can customize the inference by adding arguments to the above command.
