from __future__ import annotations
import abc
from pathlib import Path
from pydantic import BaseModel, ValidationError
from enum import Enum
from typing import Any
from dataclasses import dataclass, field
from pydantic.json_schema import model_json_schema

from config.config import Config


class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"


@dataclass
class FileDiff:
    path: Path
    old_content: str
    new_content: str

    is_new_file: bool = False
    is_deletion: bool = False

    def to_diff(self) -> str:
        import difflib

        old_lines = self.old_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)

        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] += "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        old_name = "/dev/null" if self.is_new_file else str(self.path)
        new_name = "/dev/null" if self.is_deletion else str(self.path)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_name,
            tofile=new_name,
        )

        return "".join(diff)


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    truncated: bool = False
    diff: FileDiff | None = None
    exit_code: int | None = None

    @classmethod
    def error_result(cls, error: str, output: str = "", **kwargs: Any):
        return cls(
            success=False,
            output=output,
            error=error,
            **kwargs,
        )

    @classmethod
    def success_result(cls, output: str, **kwargs: Any):
        return cls(
            success=True,
            output=output,
            error=None,
            **kwargs,
        )

    def to_model_output(self) -> str:
        if self.success:
            return self.output

        return f"Error: {self.error}\n\nOutput:\n{self.output}"


@dataclass
class ToolInvocation:
    params: dict[str, Any]
    cwd: Path


@dataclass
class ToolConfirmation:
    tool_name: str
    params: dict[str, Any]
    description: str

    diff: FileDiff | None = None
    affected_paths: list[Path] = field(default_factory=list)
    command: str | None = None
    is_dangerous: bool = False


class Tool(abc.ABC):
    name: str = "base_tool"
    description: str = "Base tool"
    kind: ToolKind = ToolKind.READ

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def schema(self) -> dict[str, Any] | type["BaseModel"]:
        raise NotImplementedError("Tool must define schema property or class attribute")

    @abc.abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema(**params)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    errors.append(f"Parameter '{field}': {msg}")

                return errors
            except Exception as e:
                return [str(e)]

        return []

    def is_mutating(self, params: dict[str, Any]) -> bool:
        return self.kind in {
            ToolKind.WRITE,
            ToolKind.SHELL,
            ToolKind.NETWORK,
            ToolKind.MEMORY,
        }

    async def get_confirmation(
        self, invocation: ToolInvocation
    ) -> ToolConfirmation | None:
        if not self.is_mutating(invocation.params):
            return None

        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}",
        )

    def to_openai_schema(self):
            """
            Converts the tool to the OpenAI function calling schema.
            Handles both dict schemas and Pydantic model schemas.
            """
            # 1. Get the parameters schema
            parameters = self.schema
            
            # 2. If it's a Pydantic model class (has a schema method), convert it to a dict
            if hasattr(parameters, "model_json_schema"):
                # Pydantic v2
                parameters = parameters.model_json_schema()
            elif hasattr(parameters, "schema") and callable(parameters.schema):
                # Pydantic v1
                parameters = parameters.schema()
                
            # 3. Wrap in the function structure vLLM expects
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": parameters
                }
            }