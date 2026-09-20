"""Exemplar library: complete programs used as few-shot examples, fallback, and goldens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from craftpilot.program.model import BuildProgram


@dataclass
class Exemplar:
    name: str
    description: str
    tags: list[str]
    program: BuildProgram
    path: Path

    def words(self) -> set[str]:
        text = f"{self.name} {self.description} {' '.join(self.tags)} {self.program.label}"
        return set(re.findall(r"[a-z]+", text.lower()))


def load_all(directory: Path) -> list[Exemplar]:
    out: list[Exemplar] = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict) or "program" not in data:
            continue
        program = BuildProgram.model_validate(data["program"])
        out.append(Exemplar(name=path.stem, description=str(data.get("description", "")).strip(),
                            tags=[str(t) for t in data.get("tags", [])], program=program, path=path))
    return out


def load_one(directory: Path, name: str) -> Exemplar | None:
    for e in load_all(directory):
        if e.name == name:
            return e
    return None


_STOP = {"a", "an", "the", "with", "and", "of", "in", "on", "to", "for", "build", "me", "please", "that", "it",
         "is", "has", "have", "some", "big", "small", "large", "make"}

# Feature and material words that appear in many exemplar tags. They still count, but a
# building-type word ("cathedral", "barn") must outrank them so "stained glass window" does
# not retrieve a modern house for a cathedral.
GENERIC = {"glass", "roof", "stone", "concrete", "tower", "house", "wood", "brick", "flat", "curved", "white",
           "dark", "red", "window", "windows", "steep", "tall", "wide", "hall", "gothic", "medieval", "rustic",
           "fantasy", "old", "new", "grand", "modern"}


def tag_score(query_words: set[str], tags: set[str]) -> int:
    return sum(1 if w in GENERIC else 3 for w in query_words & tags)


def retrieve(exemplars: list[Exemplar], query: str, k: int = 4) -> list[Exemplar]:
    q = set(re.findall(r"[a-z]+", query.lower())) - _STOP
    scored = []
    for e in exemplars:
        w = e.words()
        tags = {t.lower() for t in e.tags}
        score = tag_score(q, tags) + len(q & w)
        # Light stemming: match plurals.
        score += sum(1 for word in q if word.endswith("s") and word[:-1] in w)
        scored.append((score, e.name, e))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [e for s, _, e in scored[:k]]


def save(directory: Path, name: str, description: str, tags: list[str], program: BuildProgram) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "exemplar"
    path = directory / f"{safe}.yaml"
    data = {"description": description, "tags": tags, "program": program.model_dump(mode="json")}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
