#!/usr/bin/env python3
"""
Toploc Polyglot Signature Generator

Generates cryptographic signatures from transformer model activations.
Works with any transformer model using the standard transformers library.

Usage: python simple_signature.py

Output: signatures/llama_strawberry_signature.bin (258 bytes)
"""
from tqdm import tqdm
import sys
sys.path.insert(0, '.')
from toploc.commits import ProofPoly
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def build_activation_commit(activations, K=128):
    """Build polynomial signature from activations"""
    if not activations:
        return None
    
    # Use first activation tensor
    flat_view = activations[0].view(-1)
    topk_indices = flat_view.abs().topk(K).indices
    topk_values = flat_view[topk_indices]
    commit = ProofPoly.from_points(topk_indices.to(torch.int32), topk_values).to_bytes()
    return commit

def main():
    # Set seeds for maximum determinism
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    prompt = "how many r's in strawberry"
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        use_cache=False  # Disable KV cache for more determinism
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up activation hook
    saved_activations = []
    def activation_saving_hook(module, input, output):
        if isinstance(output, tuple):
            saved_activations.append(output[0].detach().clone().cpu())
        else:
            saved_activations.append(output.detach().clone().cpu())
    
    # Hook into final layer norm for Llama
    hook_handle = model.model.norm.register_forward_hook(activation_saving_hook)

    print(f"Generating response for: '{prompt}'")
    
    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract just the generated part (after the prompt)
    generated_text = full_response[len(prompt):].strip()
    
    print(f"Model response: {full_response}")
    print(f"Generated text: {generated_text}")
    
    # Build signature
    if saved_activations:
        signature = build_activation_commit(saved_activations)
        
        # Save signature
        save_dir = Path("signatures")
        save_dir.mkdir(exist_ok=True)
        savepath = save_dir / "llama_strawberry_signature.bin"
        
        with open(savepath, "wb") as f:
            f.write(signature)
        
        print(f"✓ Saved polyglot signature to {savepath}")
        print(f"Signature size: {len(signature)} bytes")
        print(f"Captured {len(saved_activations)} activation tensors")
    else:
        print("✗ No activations captured!")
    
    # Clean up
    hook_handle.remove()

if __name__ == "__main__":
    main()