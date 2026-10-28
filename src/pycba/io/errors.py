class AsciiFormatError(ValueError):
    """The file cannot be parsed as a 2D numeric table."""


class NoImageDataError(AsciiFormatError):
    """Fewer than 2 columns or 2 lines: not a 2D image (MATLAB returned [])."""


class AsciiFormatWarning(UserWarning):
    """Non-fatal oddity found while analysing/reading an ASCII file."""


class PaletteWarning(UserWarning):
    """Suspicious palette file (size not a power of two, interpolation to a longer palette)."""
