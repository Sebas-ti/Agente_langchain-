"""
Definición del Estado del Grafo (`State`) y Esquemas de Salida Estructurada (`Pydantic`).

CONCEPTO CLAVE DE LANGGRAPH:
----------------------------
En LangGraph, el `State` es la memoria compartida que viaja a través de todos los nodos.
Cada nodo recibe el estado actual y devuelve un diccionario con las claves que desea actualizar.

- Si una clave se define normalmente (ej. `topic: str`), el nuevo valor SOBRESCRIBE al anterior.
- Si una clave usa `Annotated[list[T], operator.add]`, LangGraph ACUMULA (concatena) los nuevos
  elementos a la lista existente. Esto es ideal para guardar hallazgos de múltiples iteraciones.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


# =====================================================================
# 1. Modelos Pydantic para Salidas Estructuradas (.with_structured_output)
# =====================================================================

class ResearchPlan(BaseModel):
    """Esquema estructurado que genera el Nodo Planificador (`planner_node`)."""

    research_goal: str = Field(
        description="Objetivo central y enfoque analítico de la investigación."
    )
    sub_queries: list[str] = Field(
        description="Lista de 3 consultas de búsqueda específicas y complementarias para investigar el tema.",
        min_length=1,
        max_length=5,
    )


class SearchFinding(BaseModel):
    """Representa un hallazgo o fuente individual recopilada por el Nodo Investigador."""

    query: str = Field(description="Consulta ejecutada que originó este hallazgo.")
    title: str = Field(description="Título del artículo, documentación o fuente.")
    url: str = Field(description="URL o ruta local de la fuente original.")
    snippet: str = Field(description="Resumen o fragmento clave extraído de la fuente.")


class CriticEvaluation(BaseModel):
    """
    Esquema estructurado que devuelve el Nodo Crítico (`critic_node`)
    aplicando el patrón de Reflexión / Auto-corrección (Self-Reflection).
    """

    quality_score: int = Field(
        description="Calificación de 1 a 10 sobre la profundidad y suficiencia de los hallazgos actuales.",
        ge=1,
        le=10,
    )
    is_sufficient: bool = Field(
        description="True si la información recopilada ya permite redactar un informe completo; False si faltan datos."
    )
    critique_reasoning: str = Field(
        description="Análisis crítico de las fortalezas actuales y qué brechas de información existen."
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="Si is_sufficient es False, nuevas consultas específicas para cubrir las brechas detectadas.",
    )


class ExecutiveBrief(BaseModel):
    """
    Esquema estructurado para el Nodo Sintetizador Ejecutivo (`summarizer_node`),
    creado siguiendo el skill `.agents/skills/add-langgraph-node/SKILL.md`.
    """

    english_abstract: str = Field(
        description="Resumen ejecutivo en inglés (Abstract) listo para compartir en equipos internacionales o portafolio."
    )
    key_takeaways: list[str] = Field(
        description="Lista de 3 acciones prácticas o conclusiones clave derivadas del informe.",
        min_length=1,
        max_length=5,
    )


# =====================================================================
# 2. Estado del Grafo de LangGraph (`ResearchState`)
# =====================================================================

class ResearchState(TypedDict):
    """Estado compartido que fluye entre los nodos del StateGraph."""

    # Entrada del usuario
    topic: str
    max_iterations: int
    use_mock: bool

    # Salida del Nodo Planificador (y actualizada por el Crítico si hay follow-ups)
    research_goal: str
    current_queries: list[str]

    # Acumuladores con operator.add (conservan el historial entre iteraciones)
    findings: Annotated[list[dict], operator.add]
    execution_logs: Annotated[list[str], operator.add]

    # Control del bucle de reflexión (Nodo Crítico)
    iteration: int
    quality_score: int
    is_sufficient: bool
    critique_notes: str

    # Salida final del Nodo Redactor y del Nodo Sintetizador Ejecutivo
    final_report: str
    english_abstract: str
    key_takeaways: list[str]
