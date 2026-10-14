"""
Birim test: Katman mimarisi kurallarını doğrular.

İç katmanlar (services, workers, adapters), dış sunum katmanına (api) bağımlı olmamalıdır.
Bu test, söz konusu dizinlerdeki tüm Python dosyalarını AST ile tarayarak
'audio_analyzer.api' importu yapılmadığını garanti eder.
"""

import ast
from pathlib import Path


def test_inner_layers_do_not_import_api_layer():
    project_root = Path(__file__).resolve().parent.parent.parent
    src_dir = project_root / "src" / "audio_analyzer"

    inner_dirs = [
        src_dir / "services",
        src_dir / "workers",
        src_dir / "adapters",
    ]

    violations = []

    for inner_dir in inner_dirs:
        assert inner_dir.exists(), f"Directory not found: {inner_dir}"
        for py_path in inner_dir.rglob("*.py"):
            code = py_path.read_text(encoding="utf-8")
            tree = ast.parse(code, filename=str(py_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "audio_analyzer.api" or alias.name.startswith("audio_analyzer.api."):
                            violations.append((str(py_path.relative_to(project_root)), node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module == "audio_analyzer.api" or node.module.startswith("audio_analyzer.api.")):
                        violations.append((str(py_path.relative_to(project_root)), node.lineno, node.module))

    assert not violations, f"Layering violation! Inner layers must not import api layer: {violations}"
