#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py - Punto de entrada principal de la aplicación
"""

import sys
import os

# Agregar src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# CAMBIO: Import absoluto en lugar de relativo
from detector_neumonia import App

if __name__ == "__main__":
    app = App()