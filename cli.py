"""
Interfaz de Línea de Comandos (CLI) enriquecida con Rich para ejecutar el agente en terminal.

Ejemplos de uso:
    # Modo Demo (sin requerir API key):
    .venv\\Scripts\\python.exe cli.py --topic "Agentes con LangGraph" --mock

    # Modo Real con Google Gemini:
    .venv\\Scripts\\python.exe cli.py --topic "RAG Agéntico vs RAG Tradicional" --max-iterations 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.deep_research.graph import build_research_graph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Research Agent con LangChain, LangGraph y Google Gemini."
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default="Arquitecturas Multi-Agente con LangGraph y Patrones de Reflexión",
        help="Tema sobre el cual investigar.",
    )
    parser.add_argument(
        "--max-iterations",
        "-m",
        type=int,
        default=2,
        help="Número máximo de iteraciones del ciclo Investigador <-> Crítico.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Ejecuta en Modo Demo/Educativo sin requerir GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Ruta opcional para guardar el informe final en un archivo .md.",
    )
    args = parser.parse_args()

    console.print(
        Panel.fit(
            f"[bold cyan]🧠 Deep Research Agent (LangChain + LangGraph)[/bold cyan]\n"
            f"[white]Tema:[/white] [bold yellow]{args.topic}[/bold yellow]\n"
            f"[white]Máx. Iteraciones:[/white] {args.max_iterations} | "
            f"[white]Modo Demo:[/white] {'Activado' if args.mock else 'Automático según .env'}",
            border_style="cyan",
        )
    )

    graph = build_research_graph(with_checkpointer=False)
    initial_state = {
        "topic": args.topic,
        "max_iterations": args.max_iterations,
        "use_mock": args.mock,
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

    final_report = ""
    english_abstract = ""
    key_takeaways: list[str] = []
    console.print("\n[bold green]▶ Iniciando streaming de nodos del StateGraph...[/bold green]\n")

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            logs = update.get("execution_logs", [])
            for log in logs:
                console.print(f"  {log}")
            if "final_report" in update:
                final_report = update["final_report"]
            if "english_abstract" in update:
                english_abstract = update["english_abstract"]
            if "key_takeaways" in update:
                key_takeaways = update["key_takeaways"]

    console.print("\n")
    console.print(
        Panel(
            Markdown(final_report),
            title="[bold green]📑 Informe Final Generado[/bold green]",
            border_style="green",
        )
    )

    if english_abstract or key_takeaways:
        takeaways_md = "\n".join([f"- {t}" for t in key_takeaways])
        console.print(
            Panel(
                Markdown(f"**{english_abstract}**\n\n### Conclusiones Accionables:\n{takeaways_md}"),
                title="[bold magenta]🌐 Executive Brief & Key Takeaways (Summarizer Node)[/bold magenta]",
                border_style="magenta",
            )
        )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(final_report, encoding="utf-8")
        console.print(f"\n[bold cyan]💾 Informe guardado en:[/bold cyan] {out_path.resolve()}")


if __name__ == "__main__":
    main()
