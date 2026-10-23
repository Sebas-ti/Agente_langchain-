"""
Herramientas de búsqueda web y base de conocimiento local (`Tools`) para el Agente Investigador.

CONCEPTO CLAVE DE LANGCHAIN:
----------------------------
Las herramientas (`@tool`) permiten que los agentes o nodos interactúen con el mundo exterior:
1. Base de conocimiento local (`knowledge/*.md` o `*.txt`) para Agentic RAG.
2. Búsqueda web real gratuita usando `ddgs` (DuckDuckGo Search) sin requerir clave de pago.
3. Fallback determinista (`use_mock=True`) para pruebas offline y CI/CD.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from src.deep_research.state import SearchFinding

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"


def search_local_knowledge(query: str) -> list[dict]:
    """
    Busca fragmentos relevantes dentro de la carpeta local `knowledge/` (archivos .md y .txt).
    Demuestra cómo combinar RAG local con búsqueda web dentro de un nodo de LangGraph.
    """
    if not KNOWLEDGE_DIR.exists():
        return []

    keywords = [w.lower() for w in query.split() if len(w) > 3]
    findings: list[dict] = []

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.*")):
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        content_lower = content.lower()
        if any(kw in content_lower for kw in keywords) or not keywords:
            snippet = " ".join(content.strip().splitlines()[:6])[:380]
            findings.append(
                SearchFinding(
                    query=query,
                    title=f"[Base Local RAG] {file_path.name}",
                    url=f"local://knowledge/{file_path.name}",
                    snippet=snippet,
                ).model_dump()
            )
    return findings


def run_web_search(
    query: str,
    max_results: int = 3,
    use_mock: bool = False,
) -> list[dict]:
    """
    Ejecuta una búsqueda híbrida (Base de Conocimiento Local + Web) y devuelve una lista de `SearchFinding`.
    Si `use_mock=True` o si falla la red, retorna hallazgos sintéticos de alta calidad para aprendizaje.
    """
    local_results = search_local_knowledge(query=query)[:1]

    if not use_mock:
        try:
            from ddgs import DDGS

            results = list(DDGS().text(query, max_results=max_results))
            findings: list[dict] = list(local_results)
            for item in results:
                finding = SearchFinding(
                    query=query,
                    title=item.get("title", "Fuente Web"),
                    url=item.get("href", "https://duckduckgo.com"),
                    snippet=item.get("body", "Sin descripción disponible."),
                )
                findings.append(finding.model_dump())
            if findings:
                return findings
        except Exception:
            pass

    mock_web = [
        SearchFinding(
            query=query,
            title=f"Arquitectura y Fundamentos: {query}",
            url="https://langchain-ai.github.io/langgraph/",
            snippet=(
                f"Análisis técnico sobre '{query}': Los sistemas multi-agente modernos basados en grafos "
                "de estado (StateGraph) permiten controlar ciclos de reflexión, memoria persistente "
                "(Checkpointers) y ejecución determinista de herramientas frente a cadenas lineales."
            ),
        ).model_dump(),
        SearchFinding(
            query=query,
            title=f"Patrones de Producción y Casos Reales: {query}",
            url="https://python.langchain.com/docs/introduction/",
            snippet=(
                f"En entornos de producción, '{query}' se beneficia del patrón Reflection (Planificador -> "
                "Investigador -> Crítico -> Redactor), reduciendo alucinaciones mediante validación "
                "con salidas estructuradas (Pydantic) y observabilidad de trazas."
            ),
        ).model_dump(),
    ]
    return local_results + mock_web


@tool
def web_search_tool(query: str) -> list[dict]:
    """Busca información actualizada en la web y en la base de conocimiento local sobre una consulta."""
    return run_web_search(query=query, max_results=3, use_mock=False)
