"""Scene model + CAD operations. Source of truth for CONTRACTS.md §2, §9.

A Scene is an ordered list of SceneObjects (a CSG stack) plus a material table and groups.
Every op is a pure function `Scene -> (Scene, message)`; the input scene is never mutated.
Ops validate their inputs and raise `SceneError` with an actionable message.
"""
from __future__ import annotations

import copy as _copy
import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .coretypes import Bbox
from .shapes import (
    MODIFIER_TYPES,
    ShapeError,
    modifier_summary,
    shape_local_bbox,
    shape_summary,
    validate_modifier,
    validate_shape,
)
from .transform import ANCHORS, Transform, anchor_point, rotate_point_about

OPS_TYPES = ("add", "subtract", "intersect", "paint")
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
MAX_DIMENSION = 200.0


class SceneError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------------------------
@dataclass
class SceneObject:
    id: str
    shape: Dict[str, Any]
    transform: Transform = field(default_factory=Transform)
    op: str = "add"
    material: Union[str, Dict[str, Any], None] = None
    modifiers: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    group: Optional[str] = None
    visible: bool = True

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SceneObject":
        if "id" not in d:
            raise SceneError("object needs an id")
        obj = SceneObject(
            id=str(d["id"]),
            shape=validate_shape(d.get("shape", {})),
            transform=Transform.from_dict(d.get("transform")),
            op=d.get("op", "add"),
            material=d.get("material"),
            modifiers=[validate_modifier(m) for m in d.get("modifiers", [])],
            tags=[str(t) for t in d.get("tags", [])],
            group=d.get("group"),
            visible=bool(d.get("visible", True)),
        )
        if obj.op not in OPS_TYPES:
            raise SceneError(f"op must be one of {OPS_TYPES}, got {obj.op!r}")
        if obj.shape["type"] == "block":
            obj.transform.anchor = "bottom_min"
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "shape": _copy.deepcopy(self.shape),
            "transform": self.transform.to_dict(),
            "op": self.op,
            "material": _copy.deepcopy(self.material),
            "modifiers": _copy.deepcopy(self.modifiers),
            "tags": list(self.tags),
            "group": self.group,
            "visible": self.visible,
        }

    def copy(self) -> "SceneObject":
        return SceneObject.from_dict(self.to_dict())

    # geometry helpers ------------------------------------------------------------------
    def local_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        return shape_local_bbox(self.shape)

    def anchor_point(self) -> np.ndarray:
        lo, hi = self.local_bbox()
        return anchor_point(lo, hi, self.transform.anchor)

    def base_bbox(self) -> Bbox:
        """World AABB of the base solid (no modifiers)."""
        lo, hi = self.local_bbox()
        wlo, whi = self.transform.world_bbox(lo, hi)
        return Bbox(wlo, whi)

    def material_name(self) -> str:
        if isinstance(self.material, str):
            return self.material
        if isinstance(self.material, dict):
            return self.material.get("name") or f"inline:{self.id}"
        return ""


@dataclass
class Scene:
    name: str = "scene"
    materials: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    objects: List[SceneObject] = field(default_factory=list)
    groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- (de)serialisation -----------------------------------------------------------------
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Scene":
        sc = Scene(
            name=str(d.get("name", "scene")),
            materials=_copy.deepcopy(d.get("materials", {})),
            objects=[SceneObject.from_dict(o) for o in d.get("objects", [])],
            groups={k: {"parent": v.get("parent") if isinstance(v, dict) else None} for k, v in d.get("groups", {}).items()},
            meta=_copy.deepcopy(d.get("meta", {})),
        )
        seen = set()
        for o in sc.objects:
            if o.id in seen:
                raise SceneError(f"duplicate object id {o.id!r}")
            seen.add(o.id)
        return sc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "materials": _copy.deepcopy(self.materials),
            "objects": [o.to_dict() for o in self.objects],
            "groups": _copy.deepcopy(self.groups),
            "meta": _copy.deepcopy(self.meta),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "Scene":
        return Scene.from_dict(json.loads(s))

    def copy(self) -> "Scene":
        return Scene.from_dict(self.to_dict())

    def hash(self) -> str:
        return hashlib.sha1(self.to_json().encode()).hexdigest()[:16]

    # -- lookup ---------------------------------------------------------------------------
    def has(self, oid: str) -> bool:
        return any(o.id == oid for o in self.objects)

    def index_of(self, oid: str) -> int:
        for i, o in enumerate(self.objects):
            if o.id == oid:
                return i
        raise SceneError(f"unknown object id {oid!r}. Known: {self._known_ids_hint()}")

    def get(self, oid: str) -> SceneObject:
        return self.objects[self.index_of(oid)]

    def _known_ids_hint(self) -> str:
        ids = [o.id for o in self.objects]
        if len(ids) > 12:
            return ", ".join(ids[:12]) + f", ... ({len(ids)} total)"
        return ", ".join(ids) if ids else "(scene is empty)"

    def group_members(self, gid: str, recursive: bool = True) -> List[str]:
        """Object ids in a group (including nested groups when recursive)."""
        if gid not in self.groups:
            raise SceneError(f"unknown group {gid!r}. Known groups: {', '.join(self.groups) or '(none)'}")
        groups = {gid}
        if recursive:
            changed = True
            while changed:
                changed = False
                for g, info in self.groups.items():
                    if info.get("parent") in groups and g not in groups:
                        groups.add(g)
                        changed = True
        return [o.id for o in self.objects if o.group in groups]

    def resolve_ids(self, ids: Union[str, Sequence[str], None], allow_empty: bool = False) -> List[str]:
        """Accepts an id, a list of ids, group ids, 'all', or a select query string. Returns object ids."""
        if ids is None:
            ids = "all"
        if isinstance(ids, str):
            ids = [ids]
        out: List[str] = []
        for token in ids:
            token = str(token).strip()
            if token == "all" or token == "*":
                out.extend(o.id for o in self.objects)
            elif self.has(token):
                out.append(token)
            elif token in self.groups:
                out.extend(self.group_members(token))
            elif any(ch in token for ch in ":*? "):
                out.extend(self.select(token))
            else:
                raise SceneError(f"unknown object or group {token!r}. Known objects: {self._known_ids_hint()}")
        seen = set()
        uniq = [x for x in out if not (x in seen or seen.add(x))]
        if not uniq and not allow_empty:
            raise SceneError(f"selection {ids!r} matched no objects")
        return uniq

    # -- selection -------------------------------------------------------------------------
    def select(self, query: str) -> List[str]:
        """Query language: space-separated terms are ANDed. Terms:
        tag:T  group:G  name:GLOB  material:M  op:OP  shape:TYPE  above_y:N  below_y:N
        east_of_x:N  west_of_x:N  south_of_z:N  north_of_z:N  visible:true|false  all
        A bare word is a name glob. Prefix a term with '-' to negate it.
        """
        terms = [t for t in query.strip().split() if t]
        if not terms or terms == ["all"]:
            return [o.id for o in self.objects]
        result: List[str] = []
        for o in self.objects:
            ok = True
            for term in terms:
                neg = term.startswith("-")
                t = term[1:] if neg else term
                m = self._match_term(o, t)
                if m == neg:
                    ok = False
                    break
            if ok:
                result.append(o.id)
        return result

    def _match_term(self, o: SceneObject, term: str) -> bool:
        if term == "all":
            return True
        if ":" not in term:
            return fnmatch.fnmatchcase(o.id, term)
        key, val = term.split(":", 1)
        key = key.lower()
        if key == "tag":
            return val in o.tags
        if key == "group":
            if val not in self.groups:
                return o.group == val
            return o.id in self.group_members(val)
        if key in ("name", "id"):
            return fnmatch.fnmatchcase(o.id, val)
        if key in ("material", "mat"):
            return fnmatch.fnmatchcase(o.material_name(), val)
        if key == "op":
            return o.op == val
        if key in ("shape", "type"):
            return o.shape["type"] == val
        if key == "visible":
            return o.visible == (val.lower() in ("1", "true", "yes"))
        try:
            n = float(val)
        except ValueError:
            raise SceneError(f"bad select term {term!r}")
        bb = self.object_bbox(o)
        if key == "above_y":
            return bb.hi[1] > n
        if key == "below_y":
            return bb.lo[1] < n
        if key == "east_of_x":
            return bb.lo[0] >= n
        if key == "west_of_x":
            return bb.hi[0] <= n
        if key == "south_of_z":
            return bb.lo[2] >= n
        if key == "north_of_z":
            return bb.hi[2] <= n
        raise SceneError(f"bad select term {term!r}")

    # -- bboxes ----------------------------------------------------------------------------
    def object_bbox(self, obj: Union[SceneObject, str]) -> Bbox:
        """World AABB including modifiers (array instances, mirror copies, displacement)."""
        if isinstance(obj, str):
            obj = self.get(obj)
        bb = obj.base_bbox()
        for m in obj.modifiers:
            t = m["type"]
            if t == "array":
                off = np.asarray(m["offset"], dtype=np.float64)
                n = int(m["count"])
                last = Bbox(bb.lo + off * (n - 1), bb.hi + off * (n - 1))
                bb = bb.union(last)
            elif t == "mirror":
                ax = "xyz".index(m["axis"])
                p = float(m["plane"])
                lo, hi = bb.lo.copy(), bb.hi.copy()
                lo[ax], hi[ax] = 2 * p - bb.hi[ax], 2 * p - bb.lo[ax]
                refl = Bbox(lo, hi)
                bb = refl if not m.get("keep_original", True) else bb.union(refl)
            elif t == "noise_displace":
                bb = bb.expanded(float(m["amplitude"]))
            elif t == "taper":
                s = float(m["top_scale"])
                if s > 1:
                    half = (bb.hi - bb.lo) / 2.0
                    c = bb.center
                    lo, hi = bb.lo.copy(), bb.hi.copy()
                    for ax in (0, 2):
                        lo[ax] = c[ax] - half[ax] * s
                        hi[ax] = c[ax] + half[ax] * s
                    bb = Bbox(lo, hi)
            elif t == "twist":
                half = (bb.hi - bb.lo) / 2.0
                r = float(np.hypot(half[0], half[2]))
                c = bb.center
                lo, hi = bb.lo.copy(), bb.hi.copy()
                lo[0], hi[0], lo[2], hi[2] = c[0] - r, c[0] + r, c[2] - r, c[2] + r
                bb = Bbox(lo, hi)
            elif t == "boolean" and m["op"] == "union" and self.has(m["target"]):
                bb = bb.union(self.get(m["target"]).base_bbox())
        return bb

    def bbox(self, ids: Union[str, Sequence[str], None] = None, adds_only: bool = True) -> Bbox:
        """Union bbox of the selected objects (default: every visible `add` object)."""
        sel = self.resolve_ids(ids, allow_empty=True) if ids is not None else [o.id for o in self.objects]
        bb = Bbox.empty()
        for oid in sel:
            o = self.get(oid)
            if not o.visible:
                continue
            if adds_only and ids is None and o.op != "add":
                continue
            bb = bb.union(self.object_bbox(o))
        return bb

    # -- outline ---------------------------------------------------------------------------
    def describe(self, ids: Union[str, Sequence[str], None] = None, detail: str = "outline") -> str:
        sel = set(self.resolve_ids(ids, allow_empty=True)) if ids not in (None, "all") else None
        bb = self.bbox()
        if bb.is_empty():
            head = f"scene {self.name}  (empty)"
        else:
            head = (
                f"scene {self.name}  bbox {_n(bb.lo[0])}..{_n(bb.hi[0])} x "
                f"{_n(bb.lo[1])}..{_n(bb.hi[1])} y {_n(bb.lo[2])}..{_n(bb.hi[2])} z"
            )
        objs = [o for o in self.objects if sel is None or o.id in sel]
        head += f"  ({len(self.objects)} objects, {len(self.groups)} groups, {len(self.materials)} materials)"
        lines = [head]
        if detail == "full" and self.materials:
            lines.append("materials:")
            for name, spec in self.materials.items():
                lines.append(f"  {name}: {_material_summary(spec)}")
        width = max([len(o.id) for o in objs] + [8])
        # ungrouped first, then groups in first-appearance order (nested shown as parent/child)
        order: List[Optional[str]] = [None]
        for o in objs:
            if o.group is not None and o.group not in order:
                order.append(o.group)
        for g in order:
            members = [o for o in objs if o.group == g]
            if not members:
                continue
            if g is not None:
                lines.append(f"[{self._group_path(g)}]")
            for o in members:
                lines.append("  " + self._object_line(o, width, detail))
        if len(lines) == 1:
            lines.append("  (no objects)")
        return "\n".join(lines)

    def _group_path(self, gid: str) -> str:
        path = [gid]
        seen = {gid}
        while True:
            parent = self.groups.get(path[0], {}).get("parent")
            if not parent or parent in seen:
                break
            path.insert(0, parent)
            seen.add(parent)
        return "/".join(path)

    def _object_line(self, o: SceneObject, width: int, detail: str) -> str:
        t = o.transform
        s = f"{o.id:<{width}} {shape_summary(o.shape)} @ ({_n(t.pos[0])},{_n(t.pos[1])},{_n(t.pos[2])})"
        if np.any(t.rot != 0):
            s += f" rot({_n(t.rot[0])},{_n(t.rot[1])},{_n(t.rot[2])})"
        if np.any(t.scale != 1):
            s += f" scale({_n(t.scale[0])},{_n(t.scale[1])},{_n(t.scale[2])})"
        if t.anchor != "bottom_center" and o.shape["type"] != "block":
            s += f" anchor={t.anchor}"
        if o.op == "add":
            if o.material:
                s += f" mat {o.material_name()}"
        elif o.op == "paint":
            s += f" PAINT {o.material_name()}"
        else:
            s += f" {o.op.upper()}"
        if o.modifiers:
            s += "  mods: " + " ".join(modifier_summary(m) for m in o.modifiers)
        if detail == "full":
            bb = self.object_bbox(o)
            s += f"  bbox ({_n(bb.lo[0])},{_n(bb.lo[1])},{_n(bb.lo[2])})..({_n(bb.hi[0])},{_n(bb.hi[1])},{_n(bb.hi[2])})"
            if o.tags:
                s += "  tags: " + ",".join(o.tags)
        if not o.visible:
            s += "  [hidden]"
        return s

    def stats(self) -> Dict[str, Any]:
        bb = self.bbox()
        return {
            "objects": len(self.objects),
            "groups": len(self.groups),
            "materials": len(self.materials),
            "bbox": None if bb.is_empty() else bb.to_list(),
            "size": None if bb.is_empty() else [float(v) for v in bb.size],
        }


def _n(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:.2f}".rstrip("0").rstrip(".")


def _material_summary(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        return str(spec)
    parts = []
    if "base" in spec:
        parts.append(f"base={spec['base']}")
    if "palette" in spec:
        parts.append("palette=" + "/".join(f"{p[0]}:{p[1]}" if isinstance(p, (list, tuple)) else str(p) for p in spec["palette"]))
    if "gradient" in spec and spec["gradient"]:
        g = spec["gradient"]
        parts.append(f"gradient({g.get('axis', 'y')} {g.get('from')}..{g.get('to')})")
    if "faces" in spec and spec["faces"]:
        parts.append("faces=" + ",".join(f"{k}:{v}" for k, v in spec["faces"].items() if v))
    if "fit" in spec:
        parts.append(f"fit={spec['fit']}")
    return " ".join(parts) or "{}"


# ----------------------------------------------------------------------------------------------
# Operations. Each returns (new_scene, message). `OPS` maps op name -> function(scene, **kwargs).
# ----------------------------------------------------------------------------------------------
OPS: Dict[str, Callable[..., Tuple[Scene, str]]] = {}
READ_ONLY_OPS = {"select", "describe", "bbox", "measure", "top_of", "side_of", "list_materials"}


def op(name: str):
    def deco(fn):
        OPS[name] = fn
        fn.op_name = name  # type: ignore[attr-defined]
        return fn

    return deco


def apply_op(scene: Scene, op_name: str, /, **kwargs) -> Tuple[Scene, str]:
    """Apply a named op. `op_name` is positional-only so op kwargs like `name` never clash."""
    if op_name not in OPS:
        raise SceneError(f"unknown op {op_name!r}")
    try:
        return OPS[op_name](scene, **kwargs)
    except SceneError:
        raise
    except (ShapeError, ValueError, TypeError, KeyError) as e:
        raise SceneError(f"{op_name}: {e}") from e


def _check_id(oid: Any) -> str:
    oid = str(oid)
    if not ID_RE.match(oid):
        raise SceneError(f"id {oid!r} must be snake_case: lowercase letters, digits, underscores, max 48 chars")
    if oid in ("all", "*"):
        raise SceneError(f"id {oid!r} is reserved")
    return oid


def _fmt_bbox(bb: Bbox) -> str:
    if bb.is_empty():
        return "empty"
    return f"({_n(bb.lo[0])},{_n(bb.lo[1])},{_n(bb.lo[2])})..({_n(bb.hi[0])},{_n(bb.hi[1])},{_n(bb.hi[2])})"


def _vec3(v, name="delta") -> np.ndarray:
    if isinstance(v, (int, float)):
        raise SceneError(f"{name} must be [x, y, z]")
    a = np.asarray([float(x) for x in v], dtype=np.float64)
    if a.shape != (3,):
        raise SceneError(f"{name} must be [x, y, z], got {v!r}")
    return a


def _dimension_warning(obj: SceneObject) -> str:
    size = obj.base_bbox().size
    if float(size.max()) > MAX_DIMENSION:
        return f" WARNING: {obj.id} spans {_n(size.max())} blocks on one axis (units mistake?)"
    return ""


def _material_warning(scene: Scene, material: Any) -> str:
    if material is None or isinstance(material, dict):
        return ""
    if material in scene.materials:
        return ""
    try:  # presets live in materials.py (Track 4); degrade gracefully if absent
        from .materials import PRESETS  # type: ignore

        if material in PRESETS:
            return ""
    except Exception:
        return ""
    return f" WARNING: material {material!r} is not defined (define_material or use a preset); resolver will fall back to stone"


# -- create / delete ---------------------------------------------------------------------------
@op("add")
def op_add(
    scene: Scene,
    id: str,
    shape: Dict[str, Any],
    pos=(0, 0, 0),
    rot=(0, 0, 0),
    scale=(1, 1, 1),
    anchor: str = "bottom_center",
    op: str = "add",
    material: Any = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    modifiers: Optional[List[Dict[str, Any]]] = None,
    visible: bool = True,
) -> Tuple[Scene, str]:
    sc = scene.copy()
    oid = _check_id(id)
    if sc.has(oid):
        raise SceneError(f"object {oid!r} already exists; choose another id or delete it first")
    if isinstance(rot, (int, float)):
        rot = (0, float(rot), 0)
    if op not in OPS_TYPES:
        raise SceneError(f"op must be one of {OPS_TYPES}")
    if group is not None and group not in sc.groups:
        sc.groups[group] = {"parent": None}
    obj = SceneObject.from_dict(
        {
            "id": oid,
            "shape": shape,
            "transform": {"pos": pos, "rot": rot, "scale": scale, "anchor": anchor},
            "op": op,
            "material": material,
            "modifiers": modifiers or [],
            "tags": tags or [],
            "group": group,
            "visible": visible,
        }
    )
    if op == "add" and material is None and obj.shape["type"] != "block":
        obj.material = "default"
    sc.objects.append(obj)
    bb = sc.object_bbox(obj)
    msg = f"added {oid}: {shape_summary(obj.shape)} {op} bbox {_fmt_bbox(bb)}"
    if isinstance(obj.material, dict):
        name = obj.material.get("name") or f"inline_{oid}"
        sc.materials[name] = {k: v for k, v in obj.material.items() if k != "name"}
        obj.material = name
    msg += _dimension_warning(obj) + _material_warning(sc, obj.material)
    return sc, msg


@op("paint")
def op_paint(scene: Scene, shape: Dict[str, Any], pos, material: Any, id: Optional[str] = None, rot=(0, 0, 0), scale=(1, 1, 1), anchor: str = "bottom_center", tags=None, group=None) -> Tuple[Scene, str]:
    if id is None:
        base = f"paint_{shape.get('type', 'x')}"
        n = 1
        while scene.has(f"{base}_{n}"):
            n += 1
        id = f"{base}_{n}"
    return op_add(scene, id=id, shape=shape, pos=pos, rot=rot, scale=scale, anchor=anchor, op="paint", material=material, tags=tags, group=group)


@op("delete")
def op_delete(scene: Scene, ids) -> Tuple[Scene, str]:
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    sc.objects = [o for o in sc.objects if o.id not in sel]
    for o in sc.objects:
        o.modifiers = [m for m in o.modifiers if not (m["type"] == "boolean" and m["target"] in sel)]
    used = {o.group for o in sc.objects if o.group}
    for g in list(sc.groups):
        if g not in used and not any(info.get("parent") == g for info in sc.groups.values()):
            del sc.groups[g]
    return sc, f"deleted {len(sel)}: {', '.join(sel)}"


@op("duplicate")
def op_duplicate(scene: Scene, id: str, new_id: str, offset=(0, 0, 0)) -> Tuple[Scene, str]:
    sc = scene.copy()
    src = sc.get(id)
    nid = _check_id(new_id)
    if sc.has(nid):
        raise SceneError(f"object {nid!r} already exists")
    dup = src.copy()
    dup.id = nid
    dup.transform.pos = dup.transform.pos + _vec3(offset, "offset")
    sc.objects.insert(sc.index_of(id) + 1, dup)
    return sc, f"duplicated {id} -> {nid} at ({','.join(_n(v) for v in dup.transform.pos)})"


@op("rename")
def op_rename(scene: Scene, id: str, new_id: str) -> Tuple[Scene, str]:
    sc = scene.copy()
    nid = _check_id(new_id)
    if sc.has(nid):
        raise SceneError(f"object {nid!r} already exists")
    o = sc.get(id)
    o.id = nid
    for other in sc.objects:
        for m in other.modifiers:
            if m["type"] == "boolean" and m["target"] == id:
                m["target"] = nid
    return sc, f"renamed {id} -> {nid}"


# -- transform ---------------------------------------------------------------------------------
@op("move")
def op_move(scene: Scene, ids, delta) -> Tuple[Scene, str]:
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    d = _vec3(delta)
    for oid in sel:
        sc.get(oid).transform.pos += d
    return sc, f"moved {_ids(sel)} by ({','.join(_n(v) for v in d)})"


@op("move_to")
def op_move_to(scene: Scene, ids, pos) -> Tuple[Scene, str]:
    """Move so the FIRST selected object's anchor lands at pos; others keep their relative offsets."""
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    p = _vec3(pos, "pos")
    d = p - sc.get(sel[0]).transform.pos
    for oid in sel:
        sc.get(oid).transform.pos += d
    return sc, f"moved {_ids(sel)} to ({','.join(_n(v) for v in p)})"


@op("rotate")
def op_rotate(scene: Scene, ids, deg: float, axis: str = "y", pivot: Any = "self") -> Tuple[Scene, str]:
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    axis = str(axis).lower()
    if axis not in ("x", "y", "z"):
        raise SceneError("axis must be x, y or z")
    ai = "xyz".index(axis)
    deg = float(deg)
    if pivot == "self":
        for oid in sel:
            o = sc.get(oid)
            o.transform.rot[ai] = (o.transform.rot[ai] + deg) % 360.0
        if len(sel) > 1:
            # rotate the group about the selection centre so members stay together
            bb = sc.bbox(sel)
            piv = bb.center
            piv[1] = bb.lo[1]
            for oid in sel:
                o = sc.get(oid)
                o.transform.pos = rotate_point_about(o.transform.pos, deg, axis, piv)
    else:
        if pivot == "scene":
            bb = scene.bbox()
            piv = bb.center
            piv[1] = bb.lo[1]
        else:
            piv = _vec3(pivot, "pivot")
        for oid in sel:
            o = sc.get(oid)
            o.transform.rot[ai] = (o.transform.rot[ai] + deg) % 360.0
            o.transform.pos = rotate_point_about(o.transform.pos, deg, axis, piv)
    for oid in sel:
        sc.get(oid).transform = Transform.from_dict(sc.get(oid).transform.to_dict())  # refresh matrices
    return sc, f"rotated {_ids(sel)} {_n(deg)}° about {axis}"


@op("scale")
def op_scale(scene: Scene, ids, factor, pivot: Any = "self") -> Tuple[Scene, str]:
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    f = np.asarray([factor] * 3 if isinstance(factor, (int, float)) else [float(v) for v in factor], dtype=np.float64)
    if f.shape != (3,) or np.any(f <= 0):
        raise SceneError("factor must be a positive number or [sx, sy, sz]")
    if pivot == "self":
        piv = None
    elif pivot == "scene":
        bb = scene.bbox()
        piv = bb.center
        piv[1] = bb.lo[1]
    else:
        piv = _vec3(pivot, "pivot")
    for oid in sel:
        o = sc.get(oid)
        _scale_object(o, f)
        if piv is not None:
            o.transform.pos = piv + (o.transform.pos - piv) * f
    return sc, f"scaled {_ids(sel)} by ({','.join(_n(v) for v in f)})"


def _scale_object(o: SceneObject, f: np.ndarray) -> None:
    """Scale shape params directly where the primitive supports it; else use transform.scale."""
    s = o.shape
    t = s["type"]
    uniform = bool(np.allclose(f, f[0]))
    if t in ("box", "wedge"):
        s["size"] = [s["size"][i] * f[i] for i in range(3)]
    elif t == "ellipsoid":
        s["radii"] = [s["radii"][i] * f[i] for i in range(3)]
    elif t in ("cylinder", "cone", "prism") and np.isclose(f[0], f[2]) and s.get("axis", "y") == "y":
        s["radius"] *= f[0]
        s["height"] *= f[1]
        if s.get("radius_top") is not None:
            s["radius_top"] *= f[0]
    elif t == "pyramid" and True:
        s["base"] = [s["base"][0] * f[0], s["base"][1] * f[2]]
        s["top"] = [s["top"][0] * f[0], s["top"][1] * f[2]]
        s["height"] *= f[1]
    elif t == "sphere" and uniform:
        s["radius"] *= f[0]
    elif t == "capsule" and np.isclose(f[0], f[2]):
        s["radius"] *= f[0]
        s["height"] *= f[1]
    elif t == "extrude":
        s["profile"] = [[p[0] * f[0], p[1] * f[2]] for p in s["profile"]]
        s["height"] *= f[1]
    elif t == "revolve" and np.isclose(f[0], f[2]):
        s["profile"] = [[p[0] * f[0], p[1] * f[1]] for p in s["profile"]]
    elif t == "sweep":
        s["path"] = [[p[0] * f[0], p[1] * f[1], p[2] * f[2]] for p in s["path"]]
        s["radius"] *= float(np.min(f))
    elif t == "line":
        s["from"] = [s["from"][i] * f[i] for i in range(3)]
        s["to"] = [s["to"][i] * f[i] for i in range(3)]
    elif t == "torus" and np.isclose(f[0], f[2]):
        s["major"] *= f[0]
        s["minor"] *= float(np.min(f))
    elif t in ("block", "plane_cut"):
        return
    else:
        o.transform.scale = o.transform.scale * f
        o.transform = Transform.from_dict(o.transform.to_dict())
        return
    o.shape = validate_shape(s)


@op("align")
def op_align(scene: Scene, ids, axis: str, mode: str = "min", to: Any = None) -> Tuple[Scene, str]:
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    axis = str(axis).lower()
    if axis not in ("x", "y", "z"):
        raise SceneError("axis must be x, y or z")
    ai = "xyz".index(axis)
    if mode not in ("min", "center", "max"):
        raise SceneError("mode must be min, center or max")
    if to is None:
        raise SceneError("align needs `to`: an object id or a coordinate value")
    if isinstance(to, (int, float)):
        target = float(to)
    else:
        tb = sc.object_bbox(sc.get(str(to)))
        target = {"min": tb.lo[ai], "center": tb.center[ai], "max": tb.hi[ai]}[mode]
    for oid in sel:
        o = sc.get(oid)
        bb = sc.object_bbox(o)
        cur = {"min": bb.lo[ai], "center": bb.center[ai], "max": bb.hi[ai]}[mode]
        o.transform.pos[ai] += target - cur
    return sc, f"aligned {_ids(sel)} {mode} on {axis} to {_n(target)}"


@op("stack")
def op_stack(scene: Scene, id: str, on: str, gap: float = 0.0, center: bool = True) -> Tuple[Scene, str]:
    """Place the bottom of `id` on top of `on` (optionally centring it in x/z)."""
    sc = scene.copy()
    o = sc.get(id)
    base = sc.object_bbox(sc.get(on))
    bb = sc.object_bbox(o)
    o.transform.pos[1] += (base.hi[1] + float(gap)) - bb.lo[1]
    if center:
        for ai in (0, 2):
            o.transform.pos[ai] += base.center[ai] - bb.center[ai]
    nb = sc.object_bbox(o)
    return sc, f"stacked {id} on {on}: now bbox {_fmt_bbox(nb)}"


@op("mirror_copy")
def op_mirror_copy(scene: Scene, ids, axis: str, plane: float, new_suffix: str = "_m") -> Tuple[Scene, str]:
    """Create mirrored duplicates across the plane axis=plane. Footprints mirror exactly."""
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    axis = str(axis).lower()
    if axis not in ("x", "z", "y"):
        raise SceneError("axis must be x, y or z")
    ai = "xyz".index(axis)
    plane = float(plane)
    created = []
    for oid in sel:
        src = sc.get(oid)
        nid = _check_id(oid + new_suffix)
        if sc.has(nid):
            raise SceneError(f"object {nid!r} already exists; pass a different new_suffix")
        dup = src.copy()
        dup.id = nid
        _mirror_shape_local(dup, axis)
        # mirror rotation: negate the two components orthogonal to the mirror axis
        r = dup.transform.rot
        for k in range(3):
            if k != ai:
                r[k] = (-r[k]) % 360.0
        dup.transform = Transform.from_dict(dup.transform.to_dict())
        for m in dup.modifiers:
            if m["type"] == "array":
                m["offset"][ai] = -m["offset"][ai]
            elif m["type"] == "mirror" and m["axis"] == axis:
                m["plane"] = 2 * plane - m["plane"]
        # translate so the mirrored bbox matches exactly
        old_bb = sc.object_bbox(src)
        new_lo = 2 * plane - old_bb.hi[ai]
        cur_bb = sc.object_bbox(dup)
        dup.transform.pos[ai] += new_lo - cur_bb.lo[ai]
        sc.objects.insert(sc.index_of(oid) + 1, dup)
        created.append(nid)
    return sc, f"mirrored {_ids(sel)} across {axis}={_n(plane)} -> {', '.join(created)}"


def _mirror_shape_local(o: SceneObject, axis: str) -> None:
    s = o.shape
    t = s["type"]
    ai = "xyz".index(axis)
    if t == "wedge" and axis in ("x", "z") and s["slope_axis"].lstrip("-") == axis:
        s["slope_axis"] = axis if s["slope_axis"].startswith("-") else "-" + axis
    elif t == "extrude" and axis in ("x", "z"):
        j = 0 if axis == "x" else 1
        s["profile"] = [[(-p[0] if j == 0 else p[0]), (-p[1] if j == 1 else p[1])] for p in s["profile"]]
    elif t == "sweep":
        s["path"] = [[(-p[k] if k == ai else p[k]) for k in range(3)] for p in s["path"]]
    elif t == "line":
        s["from"] = [(-s["from"][k] if k == ai else s["from"][k]) for k in range(3)]
        s["to"] = [(-s["to"][k] if k == ai else s["to"][k]) for k in range(3)]
    elif t == "plane_cut":
        s["normal"][ai] = -s["normal"][ai]
    o.shape = validate_shape(s)


# -- shape editing -----------------------------------------------------------------------------
@op("set_shape")
def op_set_shape(scene: Scene, id: str, **params) -> Tuple[Scene, str]:
    sc = scene.copy()
    o = sc.get(id)
    old_bb = sc.object_bbox(o)
    new = dict(o.shape)
    if "type" in params and params["type"] != o.shape["type"]:
        new = {"type": params["type"]}
        # carry over compatible params so `set_shape(keep, type=prism, sides=6)` just works
        carry = _carry_params(o.shape, params["type"])
        new.update(carry)
    for k, v in params.items():
        if k != "type":
            new[k] = v
    o.shape = validate_shape(new)
    if o.shape["type"] == "block":
        o.transform.anchor = "bottom_min"
    nb = sc.object_bbox(o)
    return sc, f"{id} is now {shape_summary(o.shape)} bbox {_fmt_bbox(nb)} (was {_fmt_bbox(old_bb)})" + _dimension_warning(o)


def _carry_params(old: Dict[str, Any], new_type: str) -> Dict[str, Any]:
    lo, hi = shape_local_bbox(old)
    size = hi - lo
    r = float(max(size[0], size[2]) / 2.0)
    h = float(size[1])
    out: Dict[str, Any] = {}
    if new_type in ("box", "wedge"):
        out["size"] = [float(size[0]), h, float(size[2])]
    elif new_type in ("cylinder", "cone", "prism", "capsule"):
        out["radius"] = r
        out["height"] = h
        if new_type == "prism":
            out["sides"] = old.get("sides", 6)
    elif new_type == "sphere":
        out["radius"] = r
    elif new_type == "ellipsoid":
        out["radii"] = [float(size[0] / 2), float(h / 2), float(size[2] / 2)]
    elif new_type == "pyramid":
        out["base"] = [float(size[0]), float(size[2])]
        out["height"] = h
    elif new_type == "extrude":
        sx, sz = float(size[0] / 2), float(size[2] / 2)
        out["profile"] = [[-sx, -sz], [sx, -sz], [sx, sz], [-sx, sz]]
        out["height"] = h
    return out


@op("set_op")
def op_set_op(scene: Scene, id: str, op: str) -> Tuple[Scene, str]:
    sc = scene.copy()
    if op not in OPS_TYPES:
        raise SceneError(f"op must be one of {OPS_TYPES}")
    sc.get(id).op = op
    return sc, f"{id} op = {op}"


@op("set_material")
def op_set_material(scene: Scene, ids, material: Any) -> Tuple[Scene, str]:
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    if isinstance(material, dict):
        name = material.get("name") or f"mat_{sel[0]}"
        sc.materials[name] = {k: v for k, v in material.items() if k != "name"}
        material = name
    for oid in sel:
        sc.get(oid).material = material
    return sc, f"material of {_ids(sel)} = {material}" + _material_warning(sc, material)


@op("set_anchor")
def op_set_anchor(scene: Scene, id: str, anchor: str) -> Tuple[Scene, str]:
    """Change the anchor without moving the object in the world."""
    sc = scene.copy()
    o = sc.get(id)
    if anchor not in ANCHORS:
        raise SceneError(f"anchor must be one of {ANCHORS}")
    lo, hi = o.local_bbox()
    old_a = anchor_point(lo, hi, o.transform.anchor)
    new_a = anchor_point(lo, hi, anchor)
    new_pos_world = o.transform.local_to_world(new_a[None, :], old_a)[0]
    o.transform.anchor = anchor
    o.transform.pos = new_pos_world
    o.transform = Transform.from_dict(o.transform.to_dict())
    return sc, f"{id} anchor = {anchor}, pos now ({','.join(_n(v) for v in o.transform.pos)})"


@op("set_visible")
def op_set_visible(scene: Scene, ids, visible: bool = True) -> Tuple[Scene, str]:
    sc = scene.copy()
    sel = sc.resolve_ids(ids)
    for oid in sel:
        sc.get(oid).visible = bool(visible)
    return sc, f"{_ids(sel)} visible = {bool(visible)}"


@op("reorder")
def op_reorder(scene: Scene, id: str, before: Optional[str] = None, after: Optional[str] = None) -> Tuple[Scene, str]:
    sc = scene.copy()
    if (before is None) == (after is None):
        raise SceneError("reorder needs exactly one of before= or after=")
    o = sc.objects.pop(sc.index_of(id))
    ref = before or after
    idx = sc.index_of(ref)  # type: ignore[arg-type]
    if after:
        idx += 1
    sc.objects.insert(idx, o)
    return sc, f"{id} now evaluated {'before' if before else 'after'} {ref} (position {idx})"


# -- modifiers ---------------------------------------------------------------------------------
@op("add_modifier")
def op_add_modifier(scene: Scene, id: str, modifier: Dict[str, Any], index: Optional[int] = None) -> Tuple[Scene, str]:
    sc = scene.copy()
    o = sc.get(id)
    m = validate_modifier(modifier)
    if m["type"] == "boolean" and not sc.has(m["target"]):
        raise SceneError(f"boolean target {m['target']!r} does not exist")
    if index is None:
        o.modifiers.append(m)
        index = len(o.modifiers) - 1
    else:
        o.modifiers.insert(int(index), m)
    return sc, f"{id} modifiers: {' '.join(modifier_summary(x) for x in o.modifiers)} (added at {index})"


@op("remove_modifier")
def op_remove_modifier(scene: Scene, id: str, index: int = -1) -> Tuple[Scene, str]:
    sc = scene.copy()
    o = sc.get(id)
    if not o.modifiers:
        raise SceneError(f"{id} has no modifiers")
    try:
        removed = o.modifiers.pop(int(index))
    except IndexError:
        raise SceneError(f"{id} has {len(o.modifiers)} modifiers; index {index} is out of range")
    rest = " ".join(modifier_summary(x) for x in o.modifiers) or "(none)"
    return sc, f"removed {modifier_summary(removed)} from {id}; remaining: {rest}"


@op("set_modifier")
def op_set_modifier(scene: Scene, id: str, index: int, **params) -> Tuple[Scene, str]:
    sc = scene.copy()
    o = sc.get(id)
    try:
        cur = o.modifiers[int(index)]
    except IndexError:
        raise SceneError(f"{id} has {len(o.modifiers)} modifiers; index {index} is out of range")
    new = dict(cur)
    new.update(params)
    o.modifiers[int(index)] = validate_modifier(new)
    return sc, f"{id} modifier {index} = {modifier_summary(o.modifiers[int(index)])}"


# -- grouping ----------------------------------------------------------------------------------
@op("group")
def op_group(scene: Scene, ids, group_id: str) -> Tuple[Scene, str]:
    sc = scene.copy()
    gid = _check_id(group_id)
    if sc.has(gid):
        raise SceneError(f"{gid!r} is an object id; group ids must be distinct")
    tokens = [ids] if isinstance(ids, str) else list(ids)
    child_groups = [t for t in tokens if t in sc.groups and not sc.has(t)]
    objs = sc.resolve_ids([t for t in tokens if t not in child_groups], allow_empty=True) if len(child_groups) < len(tokens) else []
    if gid not in sc.groups:
        sc.groups[gid] = {"parent": None}
    for cg in child_groups:
        if cg == gid:
            continue
        sc.groups[cg]["parent"] = gid
    for oid in objs:
        sc.get(oid).group = gid
    n = len(objs) + len(child_groups)
    return sc, f"group {gid}: {n} members ({len(objs)} objects, {len(child_groups)} sub-groups)"


@op("ungroup")
def op_ungroup(scene: Scene, group_id: str) -> Tuple[Scene, str]:
    sc = scene.copy()
    if group_id not in sc.groups:
        raise SceneError(f"unknown group {group_id!r}")
    parent = sc.groups[group_id].get("parent")
    for o in sc.objects:
        if o.group == group_id:
            o.group = parent
    for g, info in sc.groups.items():
        if info.get("parent") == group_id:
            info["parent"] = parent
    del sc.groups[group_id]
    return sc, f"ungrouped {group_id}"


# -- introspection (read-only) -----------------------------------------------------------------
@op("select")
def op_select(scene: Scene, query: str) -> Tuple[Scene, str]:
    ids = scene.select(str(query))
    return scene, (", ".join(ids) if ids else "(no objects match)")


@op("describe")
def op_describe(scene: Scene, ids=None, detail: str = "outline") -> Tuple[Scene, str]:
    return scene, scene.describe(ids, detail)


@op("bbox")
def op_bbox(scene: Scene, ids=None) -> Tuple[Scene, str]:
    bb = scene.bbox(ids)
    if bb.is_empty():
        return scene, "bbox: empty"
    s = bb.size
    return scene, f"bbox {_fmt_bbox(bb)} size {_n(s[0])}x{_n(s[1])}x{_n(s[2])} center ({','.join(_n(v) for v in bb.center)})"


@op("measure")
def op_measure(scene: Scene, id_a: str, id_b: str) -> Tuple[Scene, str]:
    a = scene.object_bbox(scene.get(id_a))
    b = scene.object_bbox(scene.get(id_b))
    parts = []
    for ai, name in enumerate("xyz"):
        gap = max(b.lo[ai] - a.hi[ai], a.lo[ai] - b.hi[ai])
        if gap >= 0:
            parts.append(f"{name}: gap {_n(gap)}")
        else:
            parts.append(f"{name}: overlap {_n(-gap)}")
    d = float(np.linalg.norm(a.center - b.center))
    return scene, f"{id_a} vs {id_b}: " + ", ".join(parts) + f"; centre distance {_n(d)}"


@op("top_of")
def op_top_of(scene: Scene, id: str) -> Tuple[Scene, str]:
    bb = scene.object_bbox(scene.get(id))
    c = bb.center
    return scene, f"top of {id}: y={_n(bb.hi[1])} centre ({_n(c[0])},{_n(bb.hi[1])},{_n(c[2])})"


@op("side_of")
def op_side_of(scene: Scene, id: str, dir: str) -> Tuple[Scene, str]:
    bb = scene.object_bbox(scene.get(id))
    c = bb.center
    d = str(dir).lower()
    table = {
        "north": ("z", bb.lo[2], (c[0], bb.lo[1], bb.lo[2])),
        "south": ("z", bb.hi[2], (c[0], bb.lo[1], bb.hi[2])),
        "east": ("x", bb.hi[0], (bb.hi[0], bb.lo[1], c[2])),
        "west": ("x", bb.lo[0], (bb.lo[0], bb.lo[1], c[2])),
        "up": ("y", bb.hi[1], (c[0], bb.hi[1], c[2])),
        "top": ("y", bb.hi[1], (c[0], bb.hi[1], c[2])),
        "down": ("y", bb.lo[1], (c[0], bb.lo[1], c[2])),
        "bottom": ("y", bb.lo[1], (c[0], bb.lo[1], c[2])),
    }
    if d not in table:
        raise SceneError("dir must be north, south, east, west, up or down")
    ax, val, pt = table[d]
    return scene, f"{d} side of {id}: {ax}={_n(val)} face centre ({','.join(_n(v) for v in pt)})"


# -- materials ---------------------------------------------------------------------------------
@op("define_material")
def op_define_material(scene: Scene, name: str, spec: Dict[str, Any]) -> Tuple[Scene, str]:
    sc = scene.copy()
    name = _check_id(name)
    if not isinstance(spec, dict):
        raise SceneError("spec must be an object like {'base': 'stone_bricks', 'palette': [...]}")
    spec = _copy.deepcopy(spec)
    if "base" not in spec:
        pal = spec.get("palette")
        if pal:
            first = pal[0]
            spec["base"] = first[0] if isinstance(first, (list, tuple)) else first
        elif "preset" not in spec:
            raise SceneError("material spec needs a 'base' block (or a palette)")
    try:  # deep validation lives in materials.py (Track 4)
        from .materials import validate_material_spec  # type: ignore

        spec = validate_material_spec(spec)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        raise SceneError(f"invalid material spec: {e}")
    verb = "updated" if name in sc.materials else "defined"
    sc.materials[name] = spec
    return sc, f"{verb} material {name}: {_material_summary(spec)}"


@op("list_materials")
def op_list_materials(scene: Scene) -> Tuple[Scene, str]:
    if not scene.materials:
        return scene, "no materials defined (presets are available: see materials catalog)"
    used: Dict[str, int] = {}
    for o in scene.objects:
        used[o.material_name()] = used.get(o.material_name(), 0) + 1
    lines = [f"{name}: {_material_summary(spec)}  (used by {used.get(name, 0)})" for name, spec in scene.materials.items()]
    return scene, "\n".join(lines)


def _ids(sel: List[str]) -> str:
    if len(sel) <= 4:
        return ", ".join(sel)
    return f"{', '.join(sel[:3])} (+{len(sel) - 3} more)"


# ----------------------------------------------------------------------------------------------
# Scripting facade: `scene.add(...)` style API bound to a mutable holder (used by run_script).
# ----------------------------------------------------------------------------------------------
class SceneEditor:
    """Mutable façade over the pure ops. Every call records (op, kwargs, message) in `.log`."""

    def __init__(self, scene: Scene, on_op: Optional[Callable[[str, Dict[str, Any], str], None]] = None):
        self.scene = scene
        self.log: List[Tuple[str, Dict[str, Any], str]] = []
        self._on_op = on_op

    def __getattr__(self, name: str):
        if name in OPS:

            def call(*args, **kwargs):
                fn = OPS[name]
                if args:
                    import inspect

                    params = [p for p in inspect.signature(fn).parameters if p != "scene"]
                    for i, a in enumerate(args):
                        if i < len(params) and params[i] != "params":
                            kwargs[params[i]] = a
                new_scene, msg = apply_op(self.scene, name, **kwargs)
                self.scene = new_scene
                self.log.append((name, kwargs, msg))
                if self._on_op:
                    self._on_op(name, kwargs, msg)
                return msg

            return call
        raise AttributeError(name)

    def ids(self) -> List[str]:
        return [o.id for o in self.scene.objects]
