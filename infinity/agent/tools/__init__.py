from .base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult


def default_registry() -> ToolRegistry:
    from . import fs, git, shell

    reg = ToolRegistry()
    fs.register(reg)
    shell.register(reg)
    git.register(reg)
    return reg


__all__ = ["Tool", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "default_registry"]
