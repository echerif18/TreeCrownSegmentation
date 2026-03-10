from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db_tree_crowns"
EMBED_MODEL = "mistral"


def _try_import_rag_stack():
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return {
        "Chroma": Chroma,
        "PyPDFLoader": PyPDFLoader,
        "StrOutputParser": StrOutputParser,
        "PromptTemplate": PromptTemplate,
        "RunnablePassthrough": RunnablePassthrough,
        "OllamaEmbeddings": OllamaEmbeddings,
        "OllamaLLM": OllamaLLM,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    }


def build_vectorstore(docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR, model: str = EMBED_MODEL) -> str:
    stack = _try_import_rag_stack()
    pdf_paths = list(Path(docs_dir).glob("**/*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")

    all_chunks = []
    for pdf_path in pdf_paths:
        loader = stack["PyPDFLoader"](str(pdf_path))
        pages = loader.load_and_split()
        splitter = stack["RecursiveCharacterTextSplitter"](chunk_size=1000, chunk_overlap=200)
        all_chunks.extend(splitter.split_documents(pages))

    embeddings = stack["OllamaEmbeddings"](model=model)
    stack["Chroma"].from_documents(
        all_chunks,
        embeddings,
        persist_directory=str(chroma_dir),
    )
    return f"Indexed {len(all_chunks)} chunks into {chroma_dir}"


def _rule_based_report(tree_cover_pct: float, location: str | None, year: str | None, compare_pct: float | None) -> str:
    if tree_cover_pct < 10:
        health = "sparse/open habitat"
    elif tree_cover_pct < 30:
        health = "open woodland or disturbed forest"
    elif tree_cover_pct < 60:
        health = "moderate forest"
    elif tree_cover_pct < 80:
        health = "dense forest"
    else:
        health = "closed canopy mature forest"

    agb = tree_cover_pct * 2.5
    carbon = agb * 0.47
    delta_txt = ""
    if compare_pct is not None:
        delta = tree_cover_pct - compare_pct
        delta_txt = f"\n- Change vs previous: {delta:+.1f}% (previous={compare_pct:.1f}%)"

    return f"""## Tree Crown Ecological Report
- Site: {location or "N/A"}
- Year: {year or "N/A"}
- Tree cover: {tree_cover_pct:.1f}%
- Health interpretation: {health}{delta_txt}

### Biomass proxy
- Above-ground biomass (proxy): ~{agb:.1f} t/ha
- Carbon stock (proxy): ~{carbon:.1f} tC/ha

### Recommendations
- Validate with field plots and temporal revisits.
- Use uncertainty-aware thresholds for management decisions.
"""


def generate_report(
    tree_cover_pct: float,
    location: str | None = None,
    year: str | None = None,
    compare_pct: float | None = None,
    ollama_model: str = "mistral",
    use_rag: bool = True,
    k: int = 5,
) -> str:
    if not use_rag:
        return _rule_based_report(tree_cover_pct, location, year, compare_pct)
    if not CHROMA_DIR.exists():
        return "RAG index not found. Build it first with `python app/build_vectorstore.py`.\n\n" + _rule_based_report(
            tree_cover_pct, location, year, compare_pct
        )
    try:
        stack = _try_import_rag_stack()
        embeddings = stack["OllamaEmbeddings"](model=ollama_model)
        vectorstore = stack["Chroma"](persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

        change_text = ""
        if compare_pct is not None:
            change_text = f" Previous tree cover was {compare_pct:.1f}%."
        question = (
            f"Tree crown detection result is {tree_cover_pct:.1f}% at {location or 'unknown site'}"
            f" in {year or 'unknown year'}.{change_text} Provide ecological interpretation."
        )

        prompt = stack["PromptTemplate"](
            input_variables=["context", "question"],
            template=(
                "You are an expert in forest ecology and remote sensing.\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Return concise markdown with: forest health, biomass/carbon, biodiversity implications, recommendations."
            ),
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": int(k)})
        chain = (
            {"context": retriever, "question": stack["RunnablePassthrough"]()}
            | prompt
            | stack["OllamaLLM"](model=ollama_model, temperature=0.3)
            | stack["StrOutputParser"]()
        )
        return chain.invoke(question)
    except Exception as e:
        return f"RAG failed: {e}\n\n" + _rule_based_report(tree_cover_pct, location, year, compare_pct)
