from __future__ import annotations

from pathlib import Path

from rag_report import CHROMA_DIR, DOCS_DIR, EMBED_MODEL, build_vectorstore


def main() -> None:
    print("Building RAG vector store")
    print(f"Docs: {DOCS_DIR}")
    print(f"Chroma: {CHROMA_DIR}")
    print(f"Embedding model: {EMBED_MODEL}")
    pdfs = list(Path(DOCS_DIR).glob("**/*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {DOCS_DIR}")
    msg = build_vectorstore()
    print(msg)


if __name__ == "__main__":
    main()
