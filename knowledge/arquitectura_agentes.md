# Base de Conocimiento Local: Patrones de Producción en LangGraph

## 1. Patrón Reflection (Auto-Corrección)
El patrón Reflection separa la generación de contenido de su evaluación crítica. Un nodo `critic` con salida estructurada (`Pydantic`) inspecciona los hallazgos y decide mediante un `conditional_edge` si se requiere otra iteración de búsqueda o si se cumple el umbral de calidad.

## 2. Human-in-the-Loop (HITL) y Checkpointers
Mediante `MemorySaver` (o `SqliteSaver` / `PostgresSaver` en producción) y `interrupt_before`, un grafo de LangGraph puede pausar su ejecución antes de ejecutar acciones críticas, permitiendo que un supervisor humano apruebe o modifique el estado (`graph.update_state`) antes de reanudar.

## 3. Agentic RAG (Búsqueda Híbrida Local + Web)
Un agente investigador moderno combina fuentes internas verificadas (archivos Markdown/PDF corporativos en `knowledge/`) con búsqueda web en tiempo real cuando la base local necesita contexto externo actualizado.
