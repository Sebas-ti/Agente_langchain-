"""
Nodos del Grafo (`Nodes`) en LangGraph.

CONCEPTO CLAVE DE LANGGRAPH:
----------------------------
Un Nodo en LangGraph es simplemente una función de Python que:
1. Recibe el estado actual (`state: ResearchState`).
2. Ejecuta lógica (llamar a un LLM con LangChain, invocar una herramienta, validar datos).
3. Devuelve un `dict` únicamente con las claves del estado que desea actualizar.

Aquí implementamos los 5 nodos de nuestro patrón **Self-Reflective Deep Research**:
- `planner_node`: Descompone el tema en sub-preguntas usando `.with_structured_output(ResearchPlan)`.
- `researcher_node`: Ejecuta búsquedas híbridas (RAG Local + Web) y acumula hallazgos (`findings`).
- `critic_node`: Evalúa la calidad de lo investigado con `.with_structured_output(CriticEvaluation)`.
- `writer_node`: Redacta el informe final en Markdown citando las fuentes recopiladas.
- `summarizer_node`: Genera un Abstract en inglés y 3 conclusiones accionables (`ExecutiveBrief`).
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.deep_research.prompts import (
    CRITIC_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
)
from src.deep_research.state import (
    CriticEvaluation,
    ExecutiveBrief,
    ResearchPlan,
    ResearchState,
)
from src.deep_research.tools import run_web_search

load_dotenv()


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Instancia el modelo Google Gemini configurado en el entorno."""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró GOOGLE_API_KEY. Configúrala en tu archivo .env o activa el 'Modo Demo (Sin API Key)'."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def _should_use_mock(state: ResearchState) -> bool:
    """Determina si el nodo debe ejecutarse en modo simulado (sin consumir API key)."""
    if state.get("use_mock", False):
        return True
    return not bool(os.getenv("GOOGLE_API_KEY"))


# =====================================================================
# NODO 1: PLANIFICADOR (Planner)
# =====================================================================

def planner_node(state: ResearchState) -> dict:
    """
    Analiza el tema del usuario y genera un plan estructurado (`ResearchPlan`)
    con consultas iniciales de investigación.
    """
    topic = state["topic"]
    use_mock = _should_use_mock(state)

    if use_mock:
        plan = ResearchPlan(
            research_goal=f"Investigar en profundidad los fundamentos, arquitectura práctica y retos de: {topic}",
            sub_queries=[
                f"{topic} arquitectura y fundamentos técnicos",
                f"{topic} casos de uso reales y mejores prácticas en producción",
                f"{topic} comparativa, limitaciones y tendencias actuales",
            ],
        )
    else:
        llm = get_llm(temperature=0.2)
        structured_llm = llm.with_structured_output(ResearchPlan)
        plan = structured_llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=f"Tema a investigar: {topic}"),
            ]
        )

    log_entry = (
        f"🧠 [Planner] Objetivo definido: '{plan.research_goal}' | "
        f"Consultas generadas: {len(plan.sub_queries)}"
    )

    return {
        "research_goal": plan.research_goal,
        "current_queries": plan.sub_queries,
        "iteration": 0,
        "execution_logs": [log_entry],
    }


# =====================================================================
# NODO 2: INVESTIGADOR (Researcher)
# =====================================================================

def researcher_node(state: ResearchState) -> dict:
    """
    Ejecuta búsquedas híbridas (Base Local RAG + Web) para cada consulta en `current_queries`
    y acumula los resultados en `findings` (gracias a `Annotated[list, operator.add]`).
    """
    queries = state.get("current_queries", [])
    iteration = state.get("iteration", 0) + 1
    use_mock = _should_use_mock(state)

    new_findings: list[dict] = []
    for query in queries:
        results = run_web_search(query=query, max_results=2, use_mock=use_mock)
        new_findings.extend(results)

    log_entry = (
        f"🔎 [Researcher - Iteración {iteration}] Ejecutadas {len(queries)} consultas | "
        f"Nuevos hallazgos obtenidos (RAG Local + Web): {len(new_findings)}"
    )

    return {
        "findings": new_findings,
        "iteration": iteration,
        "execution_logs": [log_entry],
    }


# =====================================================================
# NODO 3: CRÍTICO / REFLEXIÓN (Critic)
# =====================================================================

def critic_node(state: ResearchState) -> dict:
    """
    Audita los hallazgos acumulados. Si detecta vacíos y aún no se alcanzó
    `max_iterations`, propone nuevas consultas (`follow_up_queries`).
    """
    topic = state["topic"]
    goal = state.get("research_goal", "")
    findings = state.get("findings", [])
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)
    use_mock = _should_use_mock(state)

    if use_mock:
        if iteration < max_iterations:
            evaluation = CriticEvaluation(
                quality_score=6,
                is_sufficient=False,
                critique_reasoning=(
                    "Los hallazgos iniciales cubren bien la teoría general, pero falta profundizar "
                    "en patrones de evaluación, observabilidad y despliegue en producción."
                ),
                follow_up_queries=[
                    f"{topic} observabilidad, evaluación con LangSmith y despliegue en producción"
                ],
            )
        else:
            evaluation = CriticEvaluation(
                quality_score=9,
                is_sufficient=True,
                critique_reasoning=(
                    "Cobertura excelente tras el ciclo de reflexión. Se cuenta con fundamentos, "
                    "arquitectura práctica, observabilidad y fuentes verificables."
                ),
                follow_up_queries=[],
            )
    else:
        llm = get_llm(temperature=0.1)
        structured_critic = llm.with_structured_output(CriticEvaluation)
        findings_summary = "\n\n".join(
            [
                f"- Consulta: {f['query']}\n  Título: {f['title']} ({f['url']})\n  Resumen: {f['snippet']}"
                for f in findings
            ]
        )
        evaluation = structured_critic.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Tema: {topic}\n"
                        f"Objetivo: {goal}\n"
                        f"Iteración actual: {iteration} de {max_iterations}\n\n"
                        f"Hallazgos recopilados ({len(findings)}):\n{findings_summary}"
                    )
                ),
            ]
        )

    log_entry = (
        f"⚖️ [Critic - Iteración {iteration}] Puntaje de calidad: {evaluation.quality_score}/10 | "
        f"¿Suficiente?: {'Sí' if evaluation.is_sufficient else 'No — requiere profundizar'}"
    )

    next_queries = evaluation.follow_up_queries if not evaluation.is_sufficient else []

    return {
        "quality_score": evaluation.quality_score,
        "is_sufficient": evaluation.is_sufficient,
        "critique_notes": evaluation.critique_reasoning,
        "current_queries": next_queries,
        "execution_logs": [log_entry],
    }


# =====================================================================
# NODO 4: REDACTOR (Writer)
# =====================================================================

def writer_node(state: ResearchState) -> dict:
    """
    Sintetiza todos los hallazgos acumulados en un informe final en Markdown.
    """
    topic = state["topic"]
    goal = state.get("research_goal", "")
    findings = state.get("findings", [])
    critique_notes = state.get("critique_notes", "")
    use_mock = _should_use_mock(state)

    if use_mock:
        sources_md = "\n".join(
            [f"- [{f['title']}]({f['url']}) — *Consulta: `{f['query']}`*" for f in findings]
        )
        report = f"""# 📑 Informe de Investigación Profunda: {topic}

> **Objetivo de Investigación:** {goal}
> **Calidad Validada por el Nodo Crítico:** {state.get("quality_score", 9)}/10 (Iteraciones: {state.get("iteration", 1)})

---

## 1. Resumen Ejecutivo
Este informe fue generado automáticamente mediante una arquitectura agéntica con **LangChain** y **LangGraph** utilizando el patrón de diseño **Self-Reflection (Planificador → Investigador → Crítico → Redactor → Sintetizador Ejecutivo)**. El estudio de **{topic}** demuestra cómo el control explícito del estado (`StateGraph`) y la validación cíclica superan a los prompts lineales tradicionales.

## 2. Arquitectura y Conceptos Clave
A partir de las {len(findings)} fuentes analizadas (combinando Base de Conocimiento Local RAG y Web) durante las iteraciones del grafo, destacan tres pilares fundamentales:
1. **Orquestación Basada en Grafos (`StateGraph`)**: Permite modelar bucles de retroalimentación donde el agente audita su propio trabajo antes de responder al usuario.
2. **Salidas Estructuradas (`Pydantic` + `.with_structured_output`)**: Garantizan que las decisiones de planificación y enrutamiento cumplan contratos de datos estrictos.
3. **Acumulación de Estado (`Annotated[list, operator.add]`)**: Preserva la trazabilidad completa de cada búsqueda y decisión tomada por el sistema.

## 3. Aplicaciones Prácticas y Mejores Prácticas
- **Separación de Responsabilidades**: Dividir un problema complejo en nodos especializados (`planner`, `researcher`, `critic`, `writer`, `summarizer`) mejora la mantenibilidad y facilita las pruebas unitarias.
- **Límites de Iteración (`max_iterations`)**: Todo ciclo en LangGraph debe contar con una condición de parada determinista en sus `conditional_edges` para evitar bucles infinitos.
- **Evaluación Continua**: {critique_notes}

## 4. Fuentes Consultadas
{sources_md}
"""
    else:
        llm = get_llm(temperature=0.3)
        findings_formatted = "\n\n".join(
            [
                f"[{idx + 1}] Título: {f['title']}\nURL: {f['url']}\nConsulta: {f['query']}\nExtracto: {f['snippet']}"
                for idx, f in enumerate(findings)
            ]
        )
        response = llm.invoke(
            [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Tema principal: {topic}\n"
                        f"Objetivo: {goal}\n"
                        f"Notas del Crítico: {critique_notes}\n\n"
                        f"Fuentes y hallazgos recopilados:\n{findings_formatted}"
                    )
                ),
            ]
        )
        report = str(response.content)

    log_entry = f"✍️ [Writer] Informe final redactado ({len(report)} caracteres)."

    return {
        "final_report": report,
        "execution_logs": [log_entry],
    }


# =====================================================================
# NODO 5: SINTETIZADOR EJECUTIVO BILINGÜE (Summarizer - Creado con Skill)
# =====================================================================

def summarizer_node(state: ResearchState) -> dict:
    """
    Nodo adicional creado siguiendo el skill `.agents/skills/add-langgraph-node/SKILL.md`.
    Genera un Abstract en inglés (`english_abstract`) y 3 conclusiones accionables (`key_takeaways`).
    """
    topic = state["topic"]
    final_report = state.get("final_report", "")
    use_mock = _should_use_mock(state)

    if use_mock:
        brief = ExecutiveBrief(
            english_abstract=(
                f"Executive Summary: This deep research report explores '{topic}' using a self-reflective "
                "LangGraph StateGraph architecture combining local RAG knowledge and web search, validated "
                "through iterative Pydantic-structured critique cycles."
            ),
            key_takeaways=[
                "Implementar StateGraph con bordes condicionales y límites max_iterations para ciclos seguros.",
                "Usar .with_structured_output(PydanticModel) en nodos de decisión para eliminar errores de parseo.",
                "Combinar búsqueda local en knowledge/ (Agentic RAG) con búsqueda web para mayor precisión.",
            ],
        )
    else:
        llm = get_llm(temperature=0.2)
        structured_summarizer = llm.with_structured_output(ExecutiveBrief)
        brief = structured_summarizer.invoke(
            [
                SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
                HumanMessage(content=f"Tema: {topic}\n\nInforme:\n{final_report}"),
            ]
        )

    log_entry = (
        f"🌐 [Summarizer] Abstract en inglés y {len(brief.key_takeaways)} conclusiones ejecutivas generadas."
    )

    return {
        "english_abstract": brief.english_abstract,
        "key_takeaways": brief.key_takeaways,
        "execution_logs": [log_entry],
    }


# =====================================================================
# ENRUTADOR CONDICIONAL (Conditional Edge)
# =====================================================================

def should_continue_research(
    state: ResearchState,
) -> Literal["researcher", "writer"]:
    """
    Función de borde condicional (`conditional_edge`) que decide después del `critic_node`:
    - Si la información es suficiente O ya alcanzamos `max_iterations` -> va a `"writer"`.
    - Si falta información y quedan iteraciones disponibles -> regresa a `"researcher"`.
    """
    is_sufficient = state.get("is_sufficient", False)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)
    follow_up_queries = state.get("current_queries", [])

    if is_sufficient or iteration >= max_iterations or not follow_up_queries:
        return "writer"
    return "researcher"
