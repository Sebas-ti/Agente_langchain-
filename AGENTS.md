# Reglas de Ingeniería del Proyecto: Deep Research Agent (LangChain & LangGraph)

Este repositorio contiene un **Agente de Investigación Profunda con Ciclo de Reflexión (Self-Reflective Deep Research Agent)** construido con **LangChain**, **LangGraph** y **Google Gemini**.

## 1. Estándares de Arquitectura en LangGraph
- **Sintaxis de Grafo**: Utiliza siempre `StateGraph`, `START` y `END` importados de `langgraph.graph`. No uses las clases obsoletas de `AgentExecutor` de versiones antiguas de LangChain.
- **Estado Tipado (`State`)**: Define el estado del grafo en `src/deep_research/state.py` usando `TypedDict` y `Annotated` con reductores explícitos (por ejemplo, `operator.add` para listas acumulativas de hallazgos o fuentes).
- **Salidas Estructuradas**: Para nodos de toma de decisiones o planificación (`planner_node`, `critic_node`), utiliza esquemas `Pydantic` (`BaseModel`, `Field`) junto con `.with_structured_output()` del modelo LLM.
- **Pureza de Nodos**: Cada nodo en `src/deep_research/nodes.py` debe recibir el estado actual `(state: ResearchState)` (y opcionalmente `config: RunnableConfig`) y devolver únicamente un `dict` con las claves del estado que desea actualizar.
- **Bordes Condicionales**: Toda lógica de enrutamiento dinámico (`should_continue`) debe estar tipada con `Literal[...]` para que LangGraph valide y dibuje correctamente el grafo en Mermaid y LangGraph Studio.

## 2. Gestión de Entorno y Dependencias (Python)
- **Entorno Virtual Obligatorio**: Nunca ejecutes `pip install` de forma global. Usa siempre el entorno virtual local `.venv` (`.venv\Scripts\python.exe` y `.venv\Scripts\pip.exe` en Windows, o `.venv/bin/python` en Linux/macOS).
- **Variables de Entorno**: Las claves de API (`GOOGLE_API_KEY`, `GEMINI_MODEL`) se cargan desde `.env` mediante `python-dotenv`. Nunca subas claves reales al repositorio; mantén actualizado `.env.example`.

## 3. Pruebas y Calidad de Código
- **Modo Demo / Offline**: El agente debe poder probarse sin consumir cuota de API ni requerir conexión si se activa el modo `mock` o durante la ejecución de `pytest` en `tests/test_graph.py`.
- **Tipado y Docstrings**: Todas las funciones de nodos, herramientas y esquemas deben incluir anotaciones de tipo (`type hints`) y docstrings claros en español explicando su rol dentro del patrón agéntico.
