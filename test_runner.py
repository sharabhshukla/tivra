#!/usr/bin/env python3
"""
Test runner that sets up mock dependencies before importing and running tests
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup mock dependencies
exec(open('/tmp/mock_dependencies.py').read())

# Now import pytest and run tests
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "tests/"]))