"""
Tests for the base extractor abstract class
"""
import pytest
from tivra.base import Extractor


def test_abstract_class_instantiation():
    """Test that the abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Extractor()


def test_abstract_methods():
    """Test that abstract methods must be implemented by subclasses."""
    
    class IncompleteExtractor(Extractor):
        pass
    
    with pytest.raises(TypeError):
        IncompleteExtractor()


def test_concrete_implementation():
    """Test that a concrete implementation can be created."""
    
    class ConcreteExtractor(Extractor):
        def no_constraints(self):
            return 5
        
        def no_vars(self):
            return 10
        
        def extract_all(self):
            return "extracted"
    
    extractor = ConcreteExtractor()
    assert extractor.no_constraints() == 5
    assert extractor.no_vars() == 10
    assert extractor.extract_all() == "extracted"


def test_missing_methods():
    """Test that concrete implementation must implement all abstract methods."""
    
    class PartialExtractor(Extractor):
        def no_constraints(self):
            return 5
        
        def no_vars(self):
            return 10
        # Missing extract_all method
    
    with pytest.raises(TypeError):
        PartialExtractor()


def test_extractor_metaclass():
    """Test that the Extractor uses ABCMeta metaclass."""
    assert Extractor.__class__.__name__ == 'ABCMeta'


def test_abstract_method_names():
    """Test that the expected abstract methods are defined."""
    abstract_methods = Extractor.__abstractmethods__
    expected_methods = {'no_constraints', 'no_vars', 'extract_all'}
    assert abstract_methods == expected_methods