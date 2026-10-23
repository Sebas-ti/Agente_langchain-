---
name: add-langgraph-node
description: Guía paso a paso para diseñar, implementar, conectar y probar un nuevo nodo o herramienta dentro del StateGraph de este proyecto LangGraph. Úsala cuando el usuario pida agregar una nueva etapa, agente especialista o herramienta al flujo de investigación.
---

# Skill: Agregar un Nuevo Nodo o Herramienta a LangGraph

Sigue este procedimiento estructurado cada vez que extiendas el grafo del proyecto `Deep Research Agent`:

## Paso 1: Actualizar el Estado (`src/deep_research/state.py`)
1. Si el nuevo nodo necesita almacenar datos adicionales en el grafo, añade el campo correspondiente en `ResearchState(TypedDict)`.
2. Si el campo debe acumular elementos entre iteraciones del bucle, usa `Annotated[list[Tipo], operator.add]`.
3. Si el nodo requiere salida estructurada del LLM, define su esquema `BaseModel` de Pydantic con descripciones claras en cada `Field(description=...)`.

## Paso 2: Crear el Prompt del Sistema (`src/deep_research/prompts.py`)
1. Define una constante con el prompt del sistema para el nuevo nodo.
2. Asegúrate de delimitar claramente su rol, formato esperado y criterios de calidad.

## Paso 3: Implementar la Función del Nodo (`src/deep_research/nodes.py`)
1. Escribe una función `nombre_node(state: ResearchState) -> dict:` que lea solo las claves necesarias de `state`.
2. Soporta tanto el flujo con LLM real (`ChatGoogleGenerativeAI`) como el modo de demostración/pruebas (`use_mock=True`) para garantizar que los tests unitarios sigan funcionando sin API key.
3. Retorna únicamente un diccionario con las claves del estado actualizadas.

## Paso 4: Conectar al Grafo (`src/deep_research/graph.py`)
1. Registra el nodo con `workflow.add_node("nombre", nombre_node)`.
2. Conecta las transiciones con `workflow.add_edge(...)` o `workflow.add_conditional_edges(...)`.

## Paso 5: Verificar con Pytest (`tests/test_graph.py`)
1. Añade o actualiza un caso de prueba en `tests/test_graph.py`.
2. Ejecuta `.venv\Scripts\pytest.exe` para validar que el grafo compila y transiciona sin errores.
