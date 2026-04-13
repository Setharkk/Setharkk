from __future__ import annotations

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

from agent.core.agent import Agent
from agent.core.planner import AgentStep

console = Console()

STEP_STYLES = {
    "think": ("bold cyan", "THINK"),
    "act": ("bold yellow", "ACT"),
    "observe": ("bold green", "OBSERVE"),
    "response": ("bold magenta", "SETHARKK"),
    "plan": ("bold blue", "PLAN"),
}


def print_step(step: AgentStep):
    style, label = STEP_STYLES.get(step.kind, ("white", step.kind.upper()))

    if step.kind == "response":
        console.print()
        console.print(Panel(
            Markdown(step.content),
            title=f"[{style}]{label}[/]",
            border_style="magenta",
            padding=(1, 2),
        ))
    elif step.kind == "act":
        console.print(f"  [{style}][{label}][/] {step.tool_name}", end="")
        if step.tool_args:
            args_short = ", ".join(f"{k}={repr(v)}" for k, v in step.tool_args.items())
            console.print(f"({args_short})", style="dim")
        else:
            console.print()
    elif step.kind == "observe":
        console.print(f"  [{style}][{label}][/] {step.content}", style="dim green")
    elif step.kind == "think":
        console.print(f"  [{style}][{label}][/] {step.content}", style="dim cyan")


def print_banner():
    banner = Text()
    banner.append("  SETHARKK  ", style="bold white on dark_red")
    banner.append("  Agent Autonome Local  ", style="dim")
    console.print()
    console.print(Panel(banner, border_style="red", padding=(0, 2)))
    console.print("  Tape ta demande. Commandes : /tools /memory /status /clear /exit /file <path>", style="dim")
    console.print()


async def chat_loop(agent: Agent):
    print_banner()

    while True:
        try:
            console.print("[bold red]>>> [/]", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Arret.[/]")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            console.print("[dim]Setharkk se deconnecte.[/]")
            break

        if user_input == "/tools":
            for s in agent.tools.list_schemas():
                console.print(f"  [yellow]{s['name']}[/] : {s['description']}")
            continue

        if user_input == "/memory":
            msgs = await agent.memory.get_messages()
            console.print(f"  [dim]Messages en session : {len(msgs)}[/]")
            for m in msgs[-5:]:
                role = m["role"]
                text = m["content"][:80]
                console.print(f"    [{role}] {text}...")
            continue

        if user_input == "/clear":
            await agent.memory.new_session()
            agent._session_initialized = False
            console.print("  [dim]Nouvelle session (rien supprime en base).[/]")
            continue

        if user_input == "/status":
            core = await agent.memory.get_core_prompt()
            console.print(f"  [dim]Core Memory:[/]\n{core}")
            tools_list = agent.tools.names()
            console.print(f"  [dim]Tools: {', '.join(tools_list)}[/]")
            console.print(f"  [dim]Session: {agent.memory._session_id}[/]")
            continue

        if user_input.startswith("/file "):
            filepath = user_input[6:].strip()
            try:
                from pathlib import Path
                user_input = Path(filepath).read_text(encoding="utf-8").strip()
                console.print(f"  [dim]Prompt charge depuis {filepath} ({len(user_input)} chars)[/]")
            except Exception as e:
                console.print(f"  [red]Erreur lecture fichier: {e}[/]")
                continue

        # Run agent
        async for step in agent.run(user_input):
            print_step(step)

        console.print()
