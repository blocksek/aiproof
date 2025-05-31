#!/usr/bin/env python3
"""
Installation Test Script

Quick test to verify that all dependencies are correctly installed
and the toploc library can be imported.

Usage: python test_installation.py
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not found")
        return False
    
    try:
        sys.path.insert(0, '.')
        from toploc.commits import ProofPoly
        print("✅ Toploc ProofPoly")
    except ImportError as e:
        print(f"❌ Toploc library not found: {e}")
        return False
    
    return True

def test_c_extensions():
    """Test that C++ extensions can be loaded"""
    print("\n🔍 Testing C++ extensions...")
    
    try:
        from toploc.C.csrc.utils import get_fp_parts
        print("✅ C++ utils (get_fp_parts)")
        return True
    except ImportError as e:
        print(f"⚠️  C++ extensions not available: {e}")
        print("   (This is OK - basic functionality will still work)")
        return False

def test_file_structure():
    """Test that required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "simple_signature.py",
        "verify_signature.py", 
        "validate_single_signature.py",
        "toploc/commits.py",
        "toploc/__init__.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    print("🧪 Toploc Installation Test")
    print("=" * 40)
    
    imports_ok = test_imports()
    c_extensions_ok = test_c_extensions()
    files_ok = test_file_structure()
    
    print("\n🎯 Test Summary:")
    print("=" * 40)
    
    if imports_ok and files_ok:
        print("✅ All core tests PASSED")
        print("✅ Ready to generate signatures!")
        
        if c_extensions_ok:
            print("✅ C++ extensions available (full validation support)")
        else:
            print("⚠️  C++ extensions not available (limited validation)")
            
        print("\n🚀 Next steps:")
        print("   1. python simple_signature.py     # Generate signature")
        print("   2. python verify_signature.py     # Verify signature")
        return True
    else:
        print("❌ Some tests FAILED")
        print("\n🔧 Troubleshooting:")
        if not imports_ok:
            print("   - Run: pip install -r requirements.txt")
        if not files_ok:
            print("   - Ensure you're in the correct directory")
            print("   - Check that all files were downloaded correctly")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)