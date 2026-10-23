"""
Suite de pruebas unitarias y de integración para el grafo de LangGraph (`tests/test_graph.py`).

Incluye pruebas para:
1. Enrutamiento condicional (`should_continue_research`).
2. Herramienta de Base de Conocimiento Local RAG (`search_local_knowledge`).
3. Ciclo completo con el nodo `summarizer_node` (creado con el skill `add-langgraph-node`).
4. Patrón Human-in-the-Loop (`interrupt_before=["researcher"]` + `graph.update_state`).

Ejecución:
    .venv\\Scripts\\pytest.exe -v
"""

from __future__ import annotations

from src.deep_research.graph import build_research_graph
from src.deep_research.nodes import should_continue_research
from src.deep_research.tools import search_local_knowledge


def test_conditional_edge_routing():
    """Verifica que la función de borde condicional enrute correctamente entre researcher y writer."""
    state_needs_more = {
        "is_sufficient": False,
        "iteration": 1,
        "max_iterations": 3,
        "current_queries": ["consulta de seguimiento"],
    }
    assert should_continue_research(state_needs_more) == "researcher"

    state_sufficient = {
        "is_sufficient": True,
        "iteration": 1,
        "max_iterations": 3,
        "current_queries": ["consulta extra"],
    }
    assert should_continue_research(state_sufficient) == "writer"

    state_max_reached = {
        "is_sufficient": False,
        "iteration": 2,
        "max_iterations": 2,
        "current_queries": ["consulta extra"],
    }
    assert should_continue_research(state_max_reached) == "writer"


def test_local_knowledge_rag_tool():
    """Verifica que la herramienta de búsqueda local encuentre documentos en `knowledge/`."""
    results = search_local_knowledge("LangGraph Reflection Checkpointers")
    assert len(results) >= 1
    assert "arquitectura_agentes.md" in results[0]["title"]


def test_full_research_graph_reflection_loop():
    """
    Verifica que el StateGraph completo ejecute el bucle de reflexión y el nodo `summarizer_node`
    (Planner -> Researcher -> Critic -> Researcher -> Critic -> Writer -> Summarizer).
    """
    graph = build_research_graph(with_checkpointer=True)
    initial_state = {
        "topic": "Sistemas Multi-Agente con LangGraph",
        "max_iterations": 2,
        "use_mock": True,
        "research_goal": "",
        "current_queries": [],
        "findings": [],
        "execution_logs": [],
        "iteration": 0,
        "quality_score": 0,
        "is_sufficient": False,
        "critique_notes": "",
        "final_report": "",
        "english_abstract": "",
        "key_takeaways": [],
    }
    config = {"configurable": {"thread_id": "test-thread-1"}}

    result = graph.invoke(initial_state, config=config)

    assert result["iteration"] == 2
    assert result["is_sufficient"] is True
    assert result["quality_score"] >= 8
    assert len(result["findings"]) > 3
    assert "Informe de Investigación Profunda" in result["final_report"]
    assert "Executive Summary" in result["english_abstract"]
    assert len(result["key_takeaways"]) == 3
    # Planner (1) + Researcher (2) + Critic (2) + Writer (1) + Summarizer (1) = 7 logs
    assert len(result["execution_logs"]) == 7


def test_human_in_the_loop_interrupt_and_resume():
    """
    Verifica el patrón Human-in-the-Loop (`interrupt_before_research=True`):
    1. El grafo se detiene después de `planner` y antes de `researcher`.
    2. Un humano modifica `current_queries` usando `graph.update_state`.
    3. El grafo reanuda la ejecución (`graph.invoke(None, config)`) con la consulta editada.
    """
    graph = build_research_graph(with_checkpointer=True, interrupt_before_research=True)
    initial_state = {
        "topic": "Human in the Loop en LangGraph",
        "max_iterations": 1,
        "use_mock": True,
        "research_goal": "",
        "current_queries": [],
        "findings": [],
        "execution_logs": [],
        "iteration": 0,
        "quality_score": 0,
        "is_sufficient": False,
        "critique_notes": "",
        "final_report": "",
        "english_abstract": "",
        "key_takeaways": [],
    }
    config = {"configurable": {"thread_id": "hitl-thread-1"}}

    # 1. Primera invocación: se pausa justo antes de 'researcher'
    paused_state = graph.invoke(initial_state, config=config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("researcher",)
    assert paused_state["iteration"] == 0

    # 2. El humano edita las consultas aprobadas en el estado del grafo
    custom_query = "Consulta personalizada aprobada por supervisor humano"
    graph.update_state(config, {"current_queries": [custom_query]})

    # 3. Reanudar la ejecución pasando `None` como entrada
    final_state = graph.invoke(None, config=config)
    assert final_state["iteration"] == 1
    assert any(f["query"] == custom_query for f in final_state["findings"])
    assert "Informe de Investigación Profunda" in final_state["final_report"]
