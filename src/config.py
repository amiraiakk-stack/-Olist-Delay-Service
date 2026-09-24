from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    values: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "Settings":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return cls(root_dir=config_path.parent.parent, values=values)

    def section(self, name: str) -> dict[str, Any]:
        return self.values.get(name, {})

    def path(self, section: str, key: str) -> Path:
        value = self.section(section)[key]
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root_dir / candidate
