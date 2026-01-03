"""Basic tests"""


def test_imports():
    """Test that key imports work"""
    import fastapi  # noqa: F401
    import numpy  # noqa: F401
    import torch  # noqa: F401

    assert True


def test_basic_math():
    """Sanity check test"""
    assert 2 + 2 == 4


def test_python_version():
    """Check Python version"""
    import sys

    assert sys.version_info >= (3, 9)
