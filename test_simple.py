#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test rápido de verificación
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Probar que todos los módulos se pueden importar"""
    modules_to_test = [
        'modulos.read_img',
        'modulos.preprocess_img', 
        'modulos.load_model',
        'modulos.integrator',
        'modulos.grad_cam'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - IMPORT OK")
        except ImportError as e:
            print(f"❌ {module_name} - IMPORT FAILED: {e}")
            return False
    
    print("🎉 Todos los imports funcionan correctamente!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)