"""Exceptions raised by the parsing layer (mapped to HTTP 422 by the API)."""

from __future__ import annotations


class ParserError(ValueError):
    """Base class for input problems detected while parsing."""


class InvalidPDFError(ParserError):
    """The payload is not a decodable base64 string or not a PDF."""


class OversizedInputError(ParserError):
    """The payload exceeds the configured size limits."""


class EmptyInputError(ParserError):
    """The payload contains no usable content."""
