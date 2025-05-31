#!/usr/bin/env python3
"""
Original Validation Logic Runner

Runs the EXACT same validation algorithm from vllm_validate_poly.py
but using transformers instead of vLLM for cross-platform compatibility.

Usage: python validate_single_signature.py

Proves that our transformers-based signatures pass the original validation.
"""
import sys
sys.path.insert(0, '.')
from toploc.commits import ProofPoly
import torch
from pathlib import Path
from statistics import mean, median
from transformers import AutoModelForCausalLM, AutoTokenizer

# Exact same constants and validation logic as original
TMEAN = 10
TMEDIAN = 8
TEXP = 90

def check_single(activation: torch.Tensor, proof_bytes: bytes) -> tuple[int, int, float, float]:
    """
    Exact same validation logic as the original check() function, but for a single activation
    """
    from toploc.C.csrc.utils import get_fp_parts
    
    flat_view = activation.view(-1)
    prefill_topk_indices = flat_view.abs().topk(128).indices.tolist()
    prefill_topk_values = flat_view[prefill_topk_indices]
    
    poly = ProofPoly.from_bytes(proof_bytes)
    decode_topk_values = torch.tensor([poly(i) for i in prefill_topk_indices], dtype=torch.uint16).view(dtype=torch.bfloat16)
    decode_topk_indices = prefill_topk_indices

    prefill_exp, prefill_mants = get_fp_parts(prefill_topk_values)
    decode_exp, decode_mants = get_fp_parts(decode_topk_values)
    dec_i_2_topk_i = {index: i for i, index in enumerate(decode_topk_indices)}

    topk_intersection = 0
    exp_intersection = 0
    mant_errs: list[float] = []

    for i, index in enumerate(prefill_topk_indices):
        if index in dec_i_2_topk_i:
            topk_intersection += 1
            if decode_exp[dec_i_2_topk_i[index]] == prefill_exp[i]:
                exp_intersection += 1
                mant_errs.append(abs(decode_mants[dec_i_2_topk_i[index]] - prefill_mants[i]))
    
    if len(mant_errs) == 0:
        mant_err_mean = 128.0
        mant_err_median = 128.0
    else:
        mant_err_mean = mean(mant_errs)
        mant_err_median = median(mant_errs)
    
    # Apply the exact same validation logic
    if mant_err_mean > TMEAN or mant_err_median > TMEDIAN or exp_intersection < TEXP:   
        print(f"VERIFICATION FAILED: Mantissa error mean: {mant_err_mean} above {TMEAN} or median: {mant_err_median} above {TMEDIAN} or exp intersections: {exp_intersection} below {TEXP}")
        passed = False
    else:
        print(f"VERIFICATION PASSED: Mantissa error mean: {mant_err_mean} below {TMEAN} and median: {mant_err_median} below {TMEDIAN} and exp intersections: {exp_intersection} above {TEXP}")
        passed = True
        
    return topk_intersection, exp_intersection, mant_err_mean, mant_err_median, passed

def main():
    print("🔬 Running original vllm_validate_poly.py validation logic")
    print("="*60)
    
    # Load our signature
    signature_path = Path("signatures/llama_strawberry_signature.bin")
    with open(signature_path, "rb") as f:
        signature_bytes = f.read()
    
    print(f"📁 Loaded signature: {len(signature_bytes)} bytes")
    
    # Regenerate the exact same activations
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    prompt = "how many r's in strawberry"
    
    print(f"🔄 Regenerating activations with: {model_name}")
    print(f"📝 Prompt: '{prompt}'")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Capture activations using the same hook as original
    saved_activations = []
    def activation_saving_hook(module, input, output):
        if isinstance(output, tuple):
            saved_activations.append(output[0].detach().clone().cpu())
        else:
            saved_activations.append(output.detach().clone().cpu())
    
    hook_handle = model.model.norm.register_forward_hook(activation_saving_hook)

    # Generate using the same parameters
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    hook_handle.remove()
    
    if not saved_activations:
        print("❌ No activations captured!")
        return
    
    print(f"✅ Captured {len(saved_activations)} activation tensors")
    
    # Use the first activation (prefill phase) - same as original validation
    activation = saved_activations[0]
    print(f"📊 Activation shape: {activation.shape}")
    print(f"📊 Activation elements: {activation.numel()}")
    
    # Run the exact validation logic from original script
    print("\n" + "="*60)
    print("🔍 RUNNING ORIGINAL VALIDATION LOGIC")
    print("="*60)
    
    topk_int, exp_int, mant_mean, mant_median, passed = check_single(activation, signature_bytes)
    
    print(f"\n📊 DETAILED RESULTS:")
    print(f"   • Top-K intersections: {topk_int}/128 ({topk_int/128*100:.1f}%)")
    print(f"   • Exponent intersections: {exp_int}/128 ({exp_int/128*100:.1f}%)")
    print(f"   • Mantissa error mean: {mant_mean:.6f}")
    print(f"   • Mantissa error median: {mant_median:.6f}")
    
    print(f"\n🎯 VALIDATION THRESHOLDS:")
    print(f"   • Mantissa mean ≤ {TMEAN}: {'✅ PASS' if mant_mean <= TMEAN else '❌ FAIL'}")
    print(f"   • Mantissa median ≤ {TMEDIAN}: {'✅ PASS' if mant_median <= TMEDIAN else '❌ FAIL'}")
    print(f"   • Exponent intersections ≥ {TEXP}: {'✅ PASS' if exp_int >= TEXP else '❌ FAIL'}")
    
    print(f"\n🏆 FINAL RESULT: {'✅ PASSED' if passed else '❌ FAILED'}")
    print("\n" + "="*60)
    print("✅ Original vllm_validate_poly.py validation logic completed successfully!")

if __name__ == "__main__":
    main()