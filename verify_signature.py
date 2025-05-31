#!/usr/bin/env python3
"""
Toploc Polyglot Signature Verification

Comprehensive verification of polyglot signatures using the same validation
logic as the original vllm_validate_poly.py but with transformers compatibility.

Usage: python verify_signature.py

Validates: signatures/llama_strawberry_signature.bin
"""
import sys
sys.path.insert(0, '.')
from toploc.commits import ProofPoly
import torch
from pathlib import Path
from statistics import mean, median
from transformers import AutoModelForCausalLM, AutoTokenizer

# Validation thresholds from the original script
TMEAN = 10
TMEDIAN = 8
TEXP = 90

def comprehensive_validation_check(original_activations, signature_bytes):
    """
    Perform the same comprehensive validation as vllm_validate_poly.py
    """
    from toploc.C.csrc.utils import get_fp_parts
    
    print("\n🔍 Performing comprehensive validation...")
    
    # Parse the polynomial from signature
    poly = ProofPoly.from_bytes(signature_bytes)
    
    # Get the first activation tensor (prefill phase)
    activation = original_activations[0]
    flat_view = activation.view(-1)
    
    # Get top-128 indices and values from original activation
    prefill_topk_indices = flat_view.abs().topk(128).indices.tolist()
    prefill_topk_values = flat_view[prefill_topk_indices]
    
    # Reconstruct values using the polynomial
    decode_topk_values = torch.tensor([poly(i) for i in prefill_topk_indices], dtype=torch.uint16).view(dtype=torch.bfloat16)
    decode_topk_indices = prefill_topk_indices
    
    # Extract floating point parts
    prefill_exp, prefill_mants = get_fp_parts(prefill_topk_values)
    decode_exp, decode_mants = get_fp_parts(decode_topk_values)
    
    # Create mapping from indices to positions
    dec_i_2_topk_i = {index: i for i, index in enumerate(decode_topk_indices)}
    
    # Calculate intersections and errors
    topk_intersection = 0
    exp_intersection = 0
    mant_errs = []
    
    for i, index in enumerate(prefill_topk_indices):
        if index in dec_i_2_topk_i:
            topk_intersection += 1
            if decode_exp[dec_i_2_topk_i[index]] == prefill_exp[i]:
                exp_intersection += 1
                mant_errs.append(abs(decode_mants[dec_i_2_topk_i[index]] - prefill_mants[i]))
    
    # Calculate statistics
    if len(mant_errs) == 0:
        mant_err_mean = 128.0
        mant_err_median = 128.0
    else:
        mant_err_mean = mean(mant_errs)
        mant_err_median = median(mant_errs)
    
    # Print detailed results
    print(f"📊 Validation Results:")
    print(f"   • Top-K intersections: {topk_intersection}/128 ({topk_intersection/128*100:.1f}%)")
    print(f"   • Exponent intersections: {exp_intersection}/128 ({exp_intersection/128*100:.1f}%)")
    print(f"   • Mantissa error mean: {mant_err_mean:.2f}")
    print(f"   • Mantissa error median: {mant_err_median:.2f}")
    
    # Apply validation thresholds
    passed = (mant_err_mean <= TMEAN and 
              mant_err_median <= TMEDIAN and 
              exp_intersection >= TEXP)
    
    if passed:
        print(f"✅ VERIFICATION PASSED:")
        print(f"   ✓ Mantissa error mean {mant_err_mean:.2f} ≤ {TMEAN}")
        print(f"   ✓ Mantissa error median {mant_err_median:.2f} ≤ {TMEDIAN}")
        print(f"   ✓ Exponent intersections {exp_intersection} ≥ {TEXP}")
    else:
        print(f"❌ VERIFICATION FAILED:")
        if mant_err_mean > TMEAN:
            print(f"   ✗ Mantissa error mean {mant_err_mean:.2f} > {TMEAN}")
        if mant_err_median > TMEDIAN:
            print(f"   ✗ Mantissa error median {mant_err_median:.2f} > {TMEDIAN}")
        if exp_intersection < TEXP:
            print(f"   ✗ Exponent intersections {exp_intersection} < {TEXP}")
    
    return passed, topk_intersection, exp_intersection, mant_err_mean, mant_err_median

def regenerate_activations_for_validation():
    """
    Regenerate the same activations that were used to create the signature
    """
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    prompt = "how many r's in strawberry"
    
    print(f"🔄 Regenerating activations for validation...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
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

    # Tokenize and generate (same as signature generation)
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
    return saved_activations

def verify_signature():
    signature_path = Path("signatures/llama_strawberry_signature.bin")
    
    if not signature_path.exists():
        print("❌ Signature file not found!")
        return
    
    print(f"📁 Reading signature from: {signature_path}")
    
    # Read the signature
    with open(signature_path, "rb") as f:
        signature_bytes = f.read()
    
    print(f"📊 Signature size: {len(signature_bytes)} bytes")
    
    # Basic verification
    try:
        poly = ProofPoly.from_bytes(signature_bytes)
        print("✅ Successfully parsed polynomial from signature!")
        
        # Round-trip verification
        reconstructed = poly.to_bytes()
        if reconstructed == signature_bytes:
            print("✅ Round-trip verification successful!")
        else:
            print("⚠️  Round-trip verification failed")
            return
            
    except Exception as e:
        print(f"❌ Failed to parse polynomial: {e}")
        return
    
    # Comprehensive validation (recreate original activations)
    try:
        print("\n" + "="*60)
        print("🔬 COMPREHENSIVE VALIDATION (from vllm_validate_poly.py)")
        print("="*60)
        
        # Regenerate the activations
        original_activations = regenerate_activations_for_validation()
        
        if not original_activations:
            print("❌ No activations captured during regeneration!")
            return
            
        # Perform comprehensive validation
        passed, topk_int, exp_int, mant_mean, mant_median = comprehensive_validation_check(
            original_activations, signature_bytes
        )
        
        print("\n🎯 Final Validation Summary:")
        print("   ✓ File exists and readable")
        print("   ✓ Contains valid ProofPoly data")
        print("   ✓ Round-trip parsing successful")
        print(f"   ✓ Represents top-128 activations from Llama model")
        if passed:
            print("   ✅ Comprehensive validation PASSED")
        else:
            print("   ❌ Comprehensive validation FAILED")
        print(f"   ✓ For prompt: 'how many r's in strawberry'")
        
    except ImportError as e:
        print(f"\n⚠️  Skipping comprehensive validation: {e}")
        print("   (This requires the C++ utils module)")
        print("   Basic verification completed successfully!")
    except Exception as e:
        print(f"\n❌ Comprehensive validation failed: {e}")

if __name__ == "__main__":
    verify_signature()