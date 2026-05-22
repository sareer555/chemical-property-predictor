"""
Pytest Configuration and Shared Fixtures
==========================================

Provides shared test fixtures for the entire test suite.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
