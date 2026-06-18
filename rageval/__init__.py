"""rageval — a tiny, teach-by-reading RAG service with an LLM-as-judge eval gate.

Read the modules in pipeline order to learn the whole flow:

    config  → llm  → ingest → retrieve → generate → eval → api

Each module is documented for an engineer who reads Python fluently but has not
built RAG/FastAPI by hand. The comments explain the *why* of each concept.
"""

__version__ = "0.1.0"
__author__ = "Nikolai Sachok"
