"""
Interfaz Web Interactiva con Streamlit para el Deep Research Agent (LangChain + LangGraph).

Ejecución:
    .venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.deep_research.graph import build_research_graph

load_dotenv()

st.set_page_config(
    page_title="Deep Research Agent | LangChain & LangGraph",
    page_icon="🧠",
    layout="wide",
)

# =====================================================================
# BARRA LATERAL: Configuración y Arquitectura
# =====================================================================
with st.sidebar:
    st.title("⚙️ Configuración del Agente")
    st.caption("Proyecto de Portafolio: **LangChain + LangGraph + Gemini**")

    env_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "🔑 Google Gemini API Key",
        value=env_key,
        type="password",
        help="Obtén tu clave gratuita en https://aistudio.google.com/apikey",
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    model_choice = st.selectbox(
        "🤖 Modelo Gemini",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0,
    )
    os.environ["GEMINI_MODEL"] = model_choice

    max_iterations = st.slider(
        "🔄 Máximo de Iteraciones (Reflexión)",
        min_value=1,
        max_value=4,
        value=2,
        help="Número máximo de ciclos Investigador ⇄ Crítico antes de redactar el informe.",
    )

    use_mock = st.toggle(
        "🧪 Modo Demo / Educativo (Sin API Key)",
        value=not bool(api_key_input),
        help="Permite probar toda la arquitectura del grafo instantáneamente sin gastar cuota de API.",
    )

    st.divider()
    st.subheader("🗺️ Arquitectura del StateGraph")
    st.markdown(
        """
        1. **🧠 Planner**: Descompone el tema en consultas (`ResearchPlan`).
        2. **🔎 Researcher**: Búsqueda Híbrida (`knowledge/` Local RAG + Web).
        3. **⚖️ Critic**: Evalúa calidad (1-10) y decide si profundizar (`Conditional Edge`).
        4. **✍️ Writer**: Redacta el informe técnico final con citas.
        5. **🌐 Summarizer**: Genera *Executive Brief* en inglés y *Key Takeaways*.
        """
    )

# =====================================================================
# PANEL PRINCIPAL
# =====================================================================
st.title("🧠 Deep Research Agent con LangChain & LangGraph")
st.markdown(
    "Agente autónomo con **ciclo de reflexión**, **RAG local híbrido** y **síntesis ejecutiva bilingüe** (`StateGraph`). "
    "Ingresa cualquier tema técnico o de negocio y observa cómo el grafo planifica, investiga, audita y redacta en tiempo real."
)

default_topic = "Arquitecturas Multi-Agente con LangGraph y Patrones de Reflexión en Producción"
topic = st.text_input(
    "🎯 ¿Qué tema deseas investigar en profundidad?",
    value=default_topic,
)

col_run, col_info = st.columns([1, 3])
with col_run:
    run_btn = st.button("🚀 Iniciar Investigación", type="primary", use_container_width=True)
with col_info:
    if use_mock:
        st.info("💡 Ejecutando en **Modo Demo / Educativo**. Ingresa tu `GOOGLE_API_KEY` en la barra lateral para usar Gemini en vivo.")
    else:
        st.success(f"⚡ Conectado a **{model_choice}** con búsqueda híbrida (RAG Local + Web) activa.")

if run_btn and topic.strip():
    graph = build_research_graph(with_checkpointer=True)
    initial_state = {
        "topic": topic.strip(),
        "max_iterations": max_iterations,
        "use_mock": use_mock,
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
    config = {"configurable": {"thread_id": "streamlit-session-1"}}

    st.subheader("📡 Ejecución del Grafo en Tiempo Real (`graph.stream`)")
    final_state = dict(initial_state)

    with st.status("Ejecutando nodos del StateGraph...", expanded=True) as status_box:
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in event.items():
                final_state.update(
                    {
                        k: (final_state.get(k, []) + v if k in ("findings", "execution_logs") else v)
                        for k, v in node_update.items()
                    }
                )
                logs = node_update.get("execution_logs", [])
                latest_log = logs[-1] if logs else f"Nodo `{node_name}` completado."
                st.write(latest_log)

                if node_name == "planner":
                    with st.expander("📋 Ver Plan de Investigación Generado", expanded=False):
                        st.markdown(f"**Objetivo:** {node_update.get('research_goal')}")
                        st.markdown("**Sub-consultas:**")
                        for q in node_update.get("current_queries", []):
                            st.markdown(f"- `{q}`")

                elif node_name == "critic":
                    score = node_update.get("quality_score", 0)
                    sufficient = node_update.get("is_sufficient", False)
                    with st.expander(
                        f"⚖️ Evaluación del Nodo Crítico (Calidad: {score}/10)", expanded=False
                    ):
                        st.markdown(f"**Razonamiento:** {node_update.get('critique_notes')}")
                        if not sufficient and node_update.get("current_queries"):
                            st.warning(
                                "🔄 Información insuficiente. Regresando al nodo `researcher` con nuevas consultas:"
                            )
                            for fq in node_update.get("current_queries", []):
                                st.markdown(f"- `{fq}`")

        status_box.update(label="✅ Investigación completada exitosamente", state="complete")

    # Métricas resumen
    m1, m2, m3 = st.columns(3)
    m1.metric("Iteraciones de Búsqueda", final_state.get("iteration", 1))
    m2.metric("Fuentes Analizadas (RAG + Web)", len(final_state.get("findings", [])))
    m3.metric("Puntaje de Calidad (Crítico)", f"{final_state.get('quality_score', 0)}/10")

    # Pestañas de resultado
    tab_report, tab_brief, tab_sources, tab_logs = st.tabs(
        [
            "📑 Informe Final",
            "🌐 Executive Brief & Takeaways",
            "🔎 Fuentes Recopiladas",
            "📜 Trazas del Grafo",
        ]
    )

    with tab_report:
        report_md = final_state.get("final_report", "")
        st.markdown(report_md)
        st.download_button(
            label="📥 Descargar Informe en Markdown (.md)",
            data=report_md,
            file_name="informe_deep_research.md",
            mime="text/markdown",
        )

    with tab_brief:
        st.subheader("🇬🇧 English Executive Abstract")
        st.info(final_state.get("english_abstract", ""))
        st.subheader("🎯 Key Actionable Takeaways")
        for idx, item in enumerate(final_state.get("key_takeaways", []), start=1):
            st.markdown(f"**{idx}.** {item}")

    with tab_sources:
        for idx, item in enumerate(final_state.get("findings", []), start=1):
            st.markdown(f"**{idx}. [{item['title']}]({item['url']})**")
            st.caption(f"Consulta: `{item['query']}`")
            st.write(item["snippet"])
            st.divider()

    with tab_logs:
        for log in final_state.get("execution_logs", []):
            st.code(log, language="text")
