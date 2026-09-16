# Llama 3.2 in a jupyter notebook

This repository contains my experiments to implement inference for Llama 3.2 models family.

From the weights downloads up to tokens output using minimal dependencies (NumPy, matplotlib...)

This is obviously incredibly slow and incomplete but allows explore easily the Llama architecture.

[1-fetch-files.ipynb](1-fetch-files.ipynb) : fetching the weights and converting them from bf16 to float32 to be able to use them on CPU

[2-tokenizer.ipynb](2-tokenizer.ipynb) : implementation of the tokenizer

[3-embeddings.ipynb](3-embeddings.ipynb) : convert the tokenizer output to embeddings

[4-attention-transformer.ipynb](4-attention-transformer.ipynb) : attention mechanism and transformer layer

[5-complete-inference.ipynb](5-complete-inference.ipynb) : complete inference on CPU

[6-complete-inference-torch.ipynb](6-complete-inference-torch.ipynb) : complete inference on GPU using PyTorch

[final-inference.py](final-inference.py) : complete inference on GPU using PyTorch as standalone python script
