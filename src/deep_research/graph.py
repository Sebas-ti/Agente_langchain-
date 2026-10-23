"""
Construcción y compilación del Grafo de Estado (`StateGraph`) en LangGraph.

CONCEPTO CLAVE DE LANGGRAPH:
----------------------------
Un `StateGraph` conecta:
1. **Nodos (`add_node`)**: Las unidades de trabajo (`planner`, `researcher`, `critic`, `writer`, `summarizer`).
2. **Bordes Normales (`add_edge`)**: Transiciones fijas de A -> B.
3. **Bordes Condicionales (`add_conditional_edges`)**: Decisiones dinámicas en tiempo de ejecución.
4. **Checkpointer (`MemorySaver`) y Human-in-the-Loop (`interrupt_before`)**: Permite pausar el grafo
   antes de ejecutar la búsqueda para que un humano revise o edite el plan de investigación.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.deep_research.nodes import (
    critic_node,
    planner_node,
    researcher_node,
    should_continue_research,
    summarizer_node,
    writer_node,
)
from src.deep_research.state import ResearchState


def build_research_graph(
    with_checkpointer: bool = False,
    interrupt_before_research: bool = False,
):
    """
    Construye y compila el grafo `Deep Research Agent`.

    Args:
        with_checkpointer: Si es True, adjunta `MemorySaver()` para persistir el estado
                           por `thread_id` entre ejecuciones.
        interrupt_before_research: Si es True (requiere `with_checkpointer=True`), activa
                                   Human-in-the-Loop pausando el grafo después de `planner`
                                   y antes de `researcher` para aprobar/editar las consultas.
    """
    workflow = StateGraph(ResearchState)

    # 1. Registrar los nodos especialistas
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("summarizer", summarizer_node)

    # 2. Definir el flujo principal (Edges)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")

    # 3. Definir el ciclo de reflexión (Conditional Edge)
    workflow.add_conditional_edges(
        "critic",
        should_continue_research,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    # 4. Síntesis ejecutiva bilingüe y cierre del flujo
    workflow.add_edge("writer", "summarizer")
    workflow.add_edge("summarizer", END)

    # 5. Compilación del grafo (con soporte opcional para Checkpointer y Human-in-the-Loop)
    if with_checkpointer or interrupt_before_research:
        checkpointer = MemorySaver()
        interrupt_nodes = ["researcher"] if interrupt_before_research else None
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_nodes,
        )

    return workflow.compile()


# Instancia exportada por defecto para compatibilidad directa con `langgraph.json` (LangGraph Studio)
research_graph = build_research_graph(with_checkpointer=False)
