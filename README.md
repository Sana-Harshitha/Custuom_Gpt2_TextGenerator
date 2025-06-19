# Custom GPT-2 Text Generator

This project implements the GPT-2 model architecture from scratch using PyTorch, including all core components such as self-attention, feed-forward networks, GELU activation, and layer normalization. It supports loading pretrained GPT-2 weights (124M) released by OpenAI (in TensorFlow format) and integrates them into the custom model. A Streamlit-based interface is provided for interactive text generation.

## Project Structure

```
Custom_GPT2_Text_Generator/
├── BackBone_gpt2/
│   ├── download_gpt2_model.py        # Downloads GPT-2 (124M) checkpoint files
│   └── model_loader.py               # Converts and loads TF weights to PyTorch model
│
├── LLCore/
│   ├── GPTModel.py                   # Main model combining all Transformer blocks
│   └── Blocks/
│       ├── Masked_Multihead_Attention.py
│       ├── FeedForward_Neural_Network.py
│       ├── GELU.py
│       ├── Layer_Normalization.py
│       └── Transformer.py            # Transformer block with residual and norm
│
├── app.py                            # Streamlit UI for interactive generation
├── Generate_Next_Token.py           # Implements top-k and temperature decoding
├── Integrating_gpt2_llm.py          # Loads pretrained weights into custom GPT
├── model_and_optimizer.pth          # Optional: saved model state
```

## Key Modules

### GPTModel (`LLCore/GPTModel.py`)

Defines the full GPT-2 model using:
- Token and positional embeddings
- Stack of Transformer blocks (`n_layers`)
- Final layer normalization
- Linear output layer projecting to vocabulary size

### Transformer Block (`Blocks/Transformer.py`)

Each block contains:
- Multi-head masked self-attention
- Feed-forward network (with GELU)
- Layer normalization and residual connections

### Attention Layer (`Masked_Multihead_Attention.py`)

Implements masked self-attention with:
- Learnable query, key, value matrices
- Causal masking using upper-triangular attention masks
- Multi-head parallel processing
- Output projection

### GELU Activation (`GELU.py`)

Implements the Gaussian Error Linear Unit (GELU) approximation used in GPT models.

### Weight Loader (`BackBone_gpt2/model_loader.py`)

- Downloads GPT-2 model files from OpenAI's public URLs
- Loads weights from TensorFlow checkpoints
- Maps and assigns them to your custom PyTorch layers with shape verification

### Generation (`Generate_Next_Token.py`)

Supports:
- Top-k sampling to restrict outputs to most probable tokens
- Temperature scaling to control randomness
- Efficient autoregressive decoding loop

### Streamlit App (`app.py`)

Runs a text-generation UI where you can:
- Enter a prompt
- Adjust `max_tokens`, `top_k`, and `temperature`
- View model-generated text output

## Usage

### Install Requirements

```bash
pip install torch tensorflow streamlit tqdm requests
```

### Run Streamlit App

```bash
streamlit run app.py
```

### Example Generation

- Prompt: `Every effort`
- Top-k: `40`
- Temperature: `1.0`

Output:
```
Every effort moves you closer to your goals if you stay consistent and focused.
```

## Notes

- Model uses the `tiktoken` tokenizer (GPT-2's original encoding).
- The architecture and decoding logic are fully implemented from scratch.
- No external transformer libraries are used (like HuggingFace or Fairseq).

## Acknowledgments

- GPT-2 model weights and tokenizer released by OpenAI
- Weight conversion approach inspired by OpenAI’s original checkpoint structure
