# Pre-train_Llama2_on_AppleM1

A. To set up the environment:

1. Clone the repo:
For example, under `~/dev`, run `git clone` like this:
```
~/dev$ git clone --recursive git@github.com:CindySun89/Pre-train_Llama2_on_AppleM1.git
```
2. Update llama2's original model.py with my own adaption for training:

```
cp model.py.for_training llama/llama/model.py
``` 

B. To train the model:

1. Run the following cmd (assume you git clone the repo under `~/dev` as done in A.1:
```
PYTHONPATH=~/dev/Pre-train_llama2_on_AppleM1/llama/ python3 next_token_prediction.py
```

2. TODO:

Describe how to collect sample code data.
