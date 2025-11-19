#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test rápido para verificar la configuración
"""

def test_quick():
    """Test básico de verificación"""
    assert 1 + 1 == 2

def test_imports():
    """Test de importaciones básicas"""
    try:
        import numpy as np
        import pytest
        print("✅ Importaciones básicas funcionan")
        assert True
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        assert False

if __name__ == "__main__":
    test_quick()
    test_imports()
    print("🎉 Tests rápidos pasaron!")
    