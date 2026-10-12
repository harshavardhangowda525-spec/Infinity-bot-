"""Project detection: language, framework, package manager, entry points.

Reads only manifest files (package.json, pyproject.toml, ...) — never secrets.
"""

from __future__ import annotations

import json
from pathlib import Path

from .permissions import IGNORED_DIRS


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def detect_project(root: str | Path) -> dict:
    root = Path(root)
    info: dict = {"name": root.name, "languages": [], "frameworks": [], "package_managers": [],
                  "entry_points": [], "test_commands": [], "instructions": [], "git": (root / ".git").exists()}

    pkg = root / "package.json"
    if pkg.exists():
        data = _read_json(pkg)
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        info["languages"].append("TypeScript" if "typescript" in deps or (root / "tsconfig.json").exists() else "JavaScript")
        for dep, name in [("next", "Next.js"), ("react", "React"), ("vue", "Vue"), ("svelte", "Svelte"),
                          ("@angular/core", "Angular"), ("express", "Express"), ("vite", "Vite"),
                          ("react-native", "React Native"), ("electron", "Electron"), ("three", "Three.js")]:
            if dep in deps:
                info["frameworks"].append(name)
        if (root / "pnpm-lock.yaml").exists():
            info["package_managers"].append("pnpm")
        elif (root / "yarn.lock").exists():
            info["package_managers"].append("yarn")
        elif (root / "bun.lockb").exists():
            info["package_managers"].append("bun")
        else:
            info["package_managers"].append("npm")
        scripts = data.get("scripts", {})
        pm = info["package_managers"][-1]
        for s in ("test", "build", "lint", "typecheck"):
            if s in scripts:
                info["test_commands"].append(f"{pm} run {s}" if s != "test" else f"{pm} test")
        if data.get("main"):
            info["entry_points"].append(data["main"])

    py_markers = [root / n for n in ("pyproject.toml", "setup.py", "requirements.txt", "Pipfile")]
    if any(p.exists() for p in py_markers) or any(root.glob("*.py")):
        info["languages"].append("Python")
        text = ""
        for p in py_markers:
            if p.exists() and p.stat().st_size < 200_000:
                text += p.read_text(errors="replace").lower()
        for dep, name in [("django", "Django"), ("flask", "Flask"), ("fastapi", "FastAPI"), ("torch", "PyTorch"),
                          ("streamlit", "Streamlit"), ("pytest", "pytest")]:
            if dep in text:
                info["frameworks"].append(name)
        if (root / "poetry.lock").exists():
            info["package_managers"].append("poetry")
        elif (root / "uv.lock").exists():
            info["package_managers"].append("uv")
        else:
            info["package_managers"].append("pip")
        if (root / "tests").is_dir() or any(root.glob("test_*.py")) or "pytest" in text:
            info["test_commands"].append("python -m pytest -q")
        for ep in ("main.py", "app.py", "manage.py", "__main__.py"):
            if (root / ep).exists():
                info["entry_points"].append(ep)

    for marker, lang, pm, test in [("Cargo.toml", "Rust", "cargo", "cargo test"), ("go.mod", "Go", "go", "go test ./..."),
                                   ("pubspec.yaml", "Dart", "pub", "flutter test"), ("pom.xml", "Java", "maven", "mvn test"),
                                   ("build.gradle", "Java/Kotlin", "gradle", "gradle test")]:
        if (root / marker).exists():
            info["languages"].append(lang)
            info["package_managers"].append(pm)
            info["test_commands"].append(test)
    if any(root.glob("*.blend")):
        info["frameworks"].append("Blender")
    if (root / "index.html").exists() and not info["languages"]:
        info["languages"].append("HTML/CSS/JavaScript")
    for name in ("INFINITY.md", "AGENTS.md", "CLAUDE.md", "README.md"):
        if (root / name).exists():
            info["instructions"].append(name)

    count = 0
    for p in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in p.relative_to(root).parts):
            continue
        if p.is_file():
            count += 1
            if count >= 5000:
                break
    info["file_count"] = count
    info["file_count_capped"] = count >= 5000
    return info
