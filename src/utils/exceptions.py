"""
Custom Exceptions
=================

Defines custom exception classes for different pipeline stages.
Enables precise error handling and logging.
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class DataCollectionError(PipelineError):
    """Raised when data collection from external sources fails."""
    pass


class DescriptorError(PipelineError):
    """Raised when molecular descriptor calculation fails."""
    pass


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""
    pass


class ModelInferenceError(PipelineError):
    """Raised when model prediction fails."""
    pass


class ValidationError(PipelineError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""
    pass


class VisualizationError(PipelineError):
    """Raised when visualization generation fails."""
    pass
