from __future__ import annotations

import asyncio
import tempfile
import sys
from pathlib import Path

from agent.tools.base import Tool


class CodeExecutorTool(Tool):
    name = "code_executor"
    description = "Execute du code Python dans un subprocess isole. Retourne stdout et stderr."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Code Python a executer"},
            "timeout": {"type": "integer", "description": "Timeout en secondes"},
        },
        "required": ["code"],
    }

    def __init__(self, config: dict | None = None) -> None:
        defaults = (config or {}).get("tools", {}).get("defaults", {})
        self._default_timeout: int = defaults.get("code_timeout", 30)

    async def execute(self, code: str, timeout: int = 0, **_) -> str:
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 0
        if timeout <= 0:
            timeout = self._default_timeout
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()
            parts = []
            if out:
                parts.append(f"STDOUT:\n{out}")
            if err:
                parts.append(f"STDERR:\n{err}")
            parts.append(f"Exit code: {proc.returncode}")
            return "\n".join(parts) if parts else "Code execute sans sortie."
        except asyncio.TimeoutError:
            proc.kill()
            return f"[ERREUR] Timeout apres {timeout}s"
        finally:
            Path(script_path).unlink(missing_ok=True)
