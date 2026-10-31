"""Custom exceptions for SR project."""


class SuperResolutionException(Exception):
    """Base exception for all SR project errors."""
    pass


class InvalidConfigError(SuperResolutionException):
    """Raised when configuration is invalid or incomplete."""
    pass


class CheckpointError(SuperResolutionException):
    """Raised when checkpoint loading/saving fails."""
    pass


class InvalidInputError(SuperResolutionException):
    """Raised when input validation fails."""
    pass


class DeviceError(SuperResolutionException):
    """Raised when device setup fails."""
    pass
