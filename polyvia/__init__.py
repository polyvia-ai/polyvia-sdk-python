"""
Polyvia Python SDK
==================

A Python SDK for the Polyvia document intelligence API and MCP server.

Quick start::

    from polyvia import Polyvia

    client = Polyvia(api_key="poly_...")

    # Ingest and wait
    result = client.ingest.file("report.pdf")
    client.ingest.wait(result.task_id)

    # Query
    print(client.query("What are the key findings?").answer)

    # Connect to the MCP server
    client.mcp.print_claude_desktop_snippet()

    # Use as agent tools (OpenAI / Anthropic / LangChain)
    tools, call = client.tools.openai()
    tools, call = client.tools.anthropic()
    tools        = client.tools.langchain()
"""

__version__ = "0.3.0"

from ._client import AsyncPolyvia, Polyvia
from ._exceptions import (
    APIError,
    AuthenticationError,
    ForbiddenError,
    IngestionError,
    IngestionTimeout,
    NotFoundError,
    PolyviaError,
    RateLimitError,
    ServiceUnavailableError,
)
from ._models import (
    BatchIngestItem,
    BatchIngestResult,
    Document,
    Group,
    IngestionStatus,
    IngestResult,
    QueryResult,
    RateLimits,
    Usage,
)
from .mcp import MCPConfig

__all__ = [
    # Clients
    "Polyvia",
    "AsyncPolyvia",
    # Exceptions
    "PolyviaError",
    "APIError",
    "AuthenticationError",
    "ForbiddenError",
    "NotFoundError",
    "RateLimitError",
    "ServiceUnavailableError",
    "IngestionError",
    "IngestionTimeout",
    # Models
    "IngestResult",
    "BatchIngestResult",
    "BatchIngestItem",
    "IngestionStatus",
    "Document",
    "Group",
    "QueryResult",
    "Usage",
    "RateLimits",
    # MCP
    "MCPConfig",
]
