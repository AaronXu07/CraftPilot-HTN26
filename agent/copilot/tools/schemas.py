"""LLM-facing tool schemas (OpenAI function-calling format). CONTRACTS.md §8.

These descriptions are the most important prompt text in the system: they carry units, sign
conventions, examples and constraints. Keep them precise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

COORDS = "Coordinates are in blocks: +x east, +y up, +z south; scene origin = ground level at the build anchor; the +z (south) side is the build's FRONT (faces the player)."

SHAPE_CATALOG = """Shape object = {"type": ..., params}. Shapes sit on y=0 and are centred in x/z (sphere/ellipsoid/torus/capsule are centred at the origin; with the default anchor bottom_center their bottom still lands on pos.y). Catalogue:
- box: size=[sx,sy,sz]  (the workhorse: walls, floors, keeps)
- cylinder: radius, height, axis="y"|"x"|"z", radius_top=null (set for a frustum). Horizontal axis + op=subtract = arched cut.
- sphere: radius, half=false (half=true keeps the upper hemisphere: domes)
- ellipsoid: radii=[rx,ry,rz]
- cone: radius, height, radius_top=0  (tower roofs, spires)
- pyramid: base=[sx,sz], height, top=[tx,tz]  (hip roofs; top>0 gives a truncated pyramid)
- wedge: size=[sx,sy,sz], slope_axis="x"|"-x"|"z"|"-z"  (gable roof halves, ramps; the high side is at the +axis end, "-x" flips it)
- prism: sides, radius, height  (octagonal/hexagonal towers)
- torus: major, minor, axis="y"
- capsule: radius, height (total)
- extrude: profile=[[x,z],...] (any closed 2D polygon, >=3 points, in local xz), height  (L/T/U plans; the second workhorse)
- revolve: profile=[[r,y],...] (closed polygon with r>=0) lathed around y  (onion domes, vases, balusters)
- sweep: radius, path=[[x,y,z],...], closed=false  (tube along a polyline: bridges, pipes, curtain walls)
- plane_cut: normal=[nx,ny,nz], offset  (keeps the half-space n.p <= offset; use with op=intersect to slice)
- block: state="minecraft:lantern[hanging=true]"  (ONE explicit block at floor(pos): lanterns, torches, banners, doors, flowers, chests)
- line: from=[x,y,z], to=[x,y,z], thickness=1  (beams, ropes, diagonal supports; endpoints are local)"""

MODIFIER_CATALOG = """Modifier object = {"type": ..., params}; the stack is applied in order like Blender:
- shell: thickness  (hollow the solid leaving walls `thickness` thick — rooms, hollow towers)
- round: radius  (round every edge)
- array: count, offset=[dx,dy,dz]  (count instances, each shifted by offset in WORLD axes: battlements, columns, windows)
- mirror: axis="x"|"y"|"z", plane=<world coord>, keep_original=true  (symmetric copy across a plane)
- taper: top_scale  (xz scale at the top relative to the bottom; 0 = pointed)
- twist: deg_per_block  (rotate the xz section as y increases)
- noise_displace: amplitude, scale=4, seed=0  (rocky/organic surfaces)
- boolean: target=<object id>, op="union"|"subtract"|"intersect"  (boolean with ONE specific object instead of the whole stack)"""

SELECT_LANG = """`ids` is an object id, a group id, "all", a list of those, or a select query. Query terms are ANDed and separated by spaces: tag:T group:G name:GLOB material:M op:add|subtract|intersect|paint shape:TYPE above_y:N below_y:N east_of_x:N west_of_x:N south_of_z:N north_of_z:N; a bare word is a name glob (e.g. "tower_*"); prefix "-" to negate."""

MATERIAL_SPEC = """Material spec object: {"base": "stone_bricks" (block id, minecraft: optional; the family used for stairs/slab fitting), "palette": [["stone_bricks",0.72],["cracked_stone_bricks",0.18],["mossy_stone_bricks",0.10]] (weighted, clumped by coherent noise), "gradient": {"axis":"y","from":0,"to":6,"palette":[["cobblestone",0.6],["mossy_cobblestone",0.4]]} (overrides the palette between two heights, blended by noise), "faces": {"top":"stone_brick_slab","side":null,"bottom":null} (per exposed face override), "fit": "none"|"slab"|"stairs"|"stairs+slab"|"walls" (which sub-voxel smoothing the resolver may use; use stairs+slab for roofs, domes, curved walls), "noise": {"scale":3,"seed":7}}. Presets (medieval_stone, spruce_timber, sandstone_desert, blackstone_dark, quartz_modern, copper_roof, slate_roof, ...) can be used directly as material names."""


def _s(desc: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": "string", "description": desc}
    d.update(extra)
    return d


def _n(desc: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": "number", "description": desc}
    d.update(extra)
    return d


def _i(desc: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": "integer", "description": desc}
    d.update(extra)
    return d


def _b(desc: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": "boolean", "description": desc}
    d.update(extra)
    return d


def _vec3(desc: str) -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": desc}


def _ids(desc: str = "Which objects. " + SELECT_LANG) -> Dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "description": desc}


def _obj(desc: str, props: Dict[str, Any] | None = None, required: Sequence[str] = ()) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": "object", "description": desc, "additionalProperties": True}
    if props:
        d["properties"] = props
    if required:
        d["required"] = list(required)
    return d


def _shape() -> Dict[str, Any]:
    return _obj("Shape object, see the catalogue in the `add` description. Example: {\"type\":\"cylinder\",\"radius\":4.5,\"height\":22}", {"type": _s("Shape type", enum=["box", "cylinder", "sphere", "ellipsoid", "cone", "pyramid", "wedge", "prism", "torus", "capsule", "extrude", "revolve", "sweep", "plane_cut", "block", "line"])}, ["type"])


def _modifier() -> Dict[str, Any]:
    return _obj("Modifier object, see the catalogue in the `add_modifier` description. Example: {\"type\":\"array\",\"count\":5,\"offset\":[3,0,0]}", {"type": _s("Modifier type", enum=["shell", "round", "array", "mirror", "taper", "twist", "noise_displace", "boolean"])}, ["type"])


def _material() -> Dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "object", "additionalProperties": True}], "description": "Material name (defined with define_material, or a preset) or an inline material spec object. " + MATERIAL_SPEC}


def tool(name: str, description: str, props: Dict[str, Any], required: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(required), "additionalProperties": False},
            "strict": False,
        },
    }


ANCHOR_ENUM = ["bottom_center", "center", "bottom_min", "top_center", "bottom_max"]

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    # ---------------------------------------------------------------- create / delete
    tool(
        "add",
        "Add a parametric solid to the scene (a new layer at the END of the CSG stack; order matters: subtract only carves what came before). "
        + COORDS
        + " `pos` is where the shape's anchor point lands (default anchor bottom_center = base centre, so builds sit on the ground). Ids must be snake_case and semantic (tower_ne, keep, gate_cut). "
        "Sizes are in blocks: integer sizes at integer positions fill whole voxels. Example: add(id='tower_ne', shape={'type':'cylinder','radius':4.5,'height':22}, pos=[34,0,4], material='stone_wall', modifiers=[{'type':'shell','thickness':1}]). "
        "For an arched doorway: add(id='gate_cut', shape={'type':'box','size':[4,6,3]}, pos=[24,0,-1], op='subtract') then add(id='gate_arch', shape={'type':'cylinder','radius':2,'height':3,'axis':'z'}, pos=[24,4,-1], op='subtract').\n"
        + SHAPE_CATALOG,
        {
            "id": _s("snake_case semantic id, unique in the scene (e.g. keep, tower_ne, gate_cut, roof_main)"),
            "shape": _shape(),
            "pos": _vec3("[x, y, z] world position of the anchor point in blocks (y=0 is ground)"),
            "rot": _vec3("Euler degrees [rx, ry, rz], applied X then Y then Z. Rotate about y for plan rotation. Default [0,0,0]. Any angle is exact (solids are implicit)."),
            "scale": _vec3("[sx, sy, sz] scale factors, default [1,1,1]. Prefer changing shape params over non-uniform scale."),
            "anchor": _s("Which point of the shape's local bbox sits at pos. Default bottom_center.", enum=ANCHOR_ENUM),
            "op": _s("add = union (default); subtract = carve from everything before it; intersect = keep only the overlap with what came before; paint = recolour existing voxels inside the shape without changing geometry.", enum=["add", "subtract", "intersect", "paint"]),
            "material": _material(),
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Free tags for selection, e.g. ['tower','exterior']"},
            "group": _s("Group id to put the object in (created if missing). Groups move/rotate together."),
            "modifiers": {"type": "array", "items": _modifier(), "description": "Initial modifier stack (see add_modifier)."},
        },
        ["id", "shape", "pos"],
    ),
    tool(
        "paint",
        "Recolour existing voxels inside a shape without changing geometry (sugar for add(op='paint')). Use for accent bands, string courses, a darker base: paint(shape={'type':'box','size':[40,2,40]}, pos=[20,0,20], material='dark_base'). Evaluated after everything added before it.",
        {"shape": _shape(), "pos": _vec3("[x, y, z] anchor position"), "material": _material(), "id": _s("Optional id (auto-generated paint_<shape>_N if omitted)"), "rot": _vec3("Euler degrees"), "anchor": _s("Anchor", enum=ANCHOR_ENUM), "tags": {"type": "array", "items": {"type": "string"}}, "group": _s("Group id")},
        ["shape", "pos", "material"],
    ),
    tool("delete", "Delete objects (also removes boolean modifiers that targeted them and empty groups).", {"ids": _ids()}, ["ids"]),
    tool("duplicate", "Copy one object under a new id, optionally offset. The copy is inserted right after the original in the CSG order.", {"id": _s("Source object id"), "new_id": _s("New snake_case id"), "offset": _vec3("[dx, dy, dz] shift in blocks (+x east, +z south), default [0,0,0]")}, ["id", "new_id"]),
    tool("rename", "Rename an object (boolean modifier targets are updated).", {"id": _s("Current id"), "new_id": _s("New snake_case id")}, ["id", "new_id"]),
    # ---------------------------------------------------------------- transform
    tool("move", "Translate objects by a delta in blocks. +x east, +y up, +z south (negative z = north). Groups move as a unit when you pass the group id.", {"ids": _ids(), "delta": _vec3("[dx, dy, dz] in blocks")}, ["ids", "delta"]),
    tool("move_to", "Move so the FIRST selected object's anchor lands at pos; the others keep their relative offsets.", {"ids": _ids(), "pos": _vec3("[x, y, z] target position for the first object's anchor")}, ["ids", "pos"]),
    tool(
        "rotate",
        "Rotate objects by deg (degrees, counter-clockwise looking down the +axis... in practice: rotate about y turns the plan; 90 turns east to south). pivot='self' rotates each object about its own anchor (a multi-object selection rotates about the selection's centre), 'scene' about the scene bbox centre, or an explicit [x,y,z].",
        {"ids": _ids(), "deg": _n("Degrees"), "axis": _s("Rotation axis, default y", enum=["x", "y", "z"]), "pivot": {"anyOf": [{"type": "string", "enum": ["self", "scene"]}, _vec3("[x,y,z] pivot point")], "description": "Pivot: 'self' (default), 'scene', or [x,y,z]"}},
        ["ids", "deg"],
    ),
    tool("scale", "Scale objects by a factor (number) or [sx, sy, sz]. Shape params are scaled directly where possible (box size, cylinder radius/height, ...). pivot as in rotate.", {"ids": _ids(), "factor": {"anyOf": [{"type": "number"}, _vec3("[sx, sy, sz]")], "description": "Uniform factor or per-axis factors (>0)"}, "pivot": {"anyOf": [{"type": "string", "enum": ["self", "scene"]}, _vec3("[x,y,z]")], "description": "Pivot: 'self' (default), 'scene', or [x,y,z]"}}, ["ids", "factor"]),
    tool("align", "Align objects' bbox min/center/max on one axis to a target: another object's same edge (to='keep') or a coordinate (to=12).", {"ids": _ids(), "axis": _s("x, y or z", enum=["x", "y", "z"]), "mode": _s("Which edge of each bbox to align: min, center or max (default min)", enum=["min", "center", "max"]), "to": {"anyOf": [{"type": "string"}, {"type": "number"}], "description": "Object id (align to its same edge) or a coordinate value"}}, ["ids", "axis", "to"]),
    tool("stack", "Put the bottom of object `id` on top of object `on` (plus an optional gap) and centre it in x/z on `on` unless center=false. Use it for roofs on towers: stack(id='tower_ne_roof', on='tower_ne').", {"id": _s("Object to move"), "on": _s("Object to stand on"), "gap": _n("Extra vertical gap in blocks (default 0)"), "center": _b("Centre in x/z on the base object (default true)")}, ["id", "on"]),
    tool("mirror_copy", "Create mirrored duplicates across the plane axis=plane (world coordinate). Footprints mirror exactly; the copies get id+new_suffix. Example: mirror_copy(ids='tower_ne', axis='x', plane=20, new_suffix='_w').", {"ids": _ids(), "axis": _s("Mirror axis: x mirrors east<->west, z mirrors north<->south", enum=["x", "y", "z"]), "plane": _n("World coordinate of the mirror plane on that axis"), "new_suffix": _s("Suffix for the new ids, default '_m'")}, ["ids", "axis", "plane"]),
    # ---------------------------------------------------------------- shape editing
    tool("set_shape", "Edit any shape parameter of an object, or change its primitive type (compatible params carry over): set_shape(id='keep', params={'type':'prism','sides':6}) or set_shape(id='tower_ne', params={'height':30}). The object stays at its position.", {"id": _s("Object id"), "params": _obj("Shape params to change; may include 'type'. See the shape catalogue in `add`.")}, ["id", "params"]),
    tool("set_op", "Change an object's CSG op.", {"id": _s("Object id"), "op": _s("add, subtract, intersect or paint", enum=["add", "subtract", "intersect", "paint"])}, ["id", "op"]),
    tool("set_material", "Assign a material (name, preset, or inline spec) to objects.", {"ids": _ids(), "material": _material()}, ["ids", "material"]),
    tool("set_anchor", "Change which point of the shape sits at pos WITHOUT moving the object in the world (pos is recomputed).", {"id": _s("Object id"), "anchor": _s("New anchor", enum=ANCHOR_ENUM)}, ["id", "anchor"]),
    tool("set_visible", "Hide/show objects (hidden objects are skipped by the rasteriser).", {"ids": _ids(), "visible": _b("true to show, false to hide")}, ["ids", "visible"]),
    tool("reorder", "Change CSG evaluation order: move `id` before or after another object. A subtract only carves objects that come BEFORE it.", {"id": _s("Object id to move"), "before": _s("Place it before this object id"), "after": _s("Place it after this object id")}, ["id"]),
    # ---------------------------------------------------------------- modifiers
    tool("add_modifier", "Append (or insert at index) a modifier on an object's ordered stack.\n" + MODIFIER_CATALOG + "\nExample battlements: add(id='merlon', shape={'type':'box','size':[1,1,1]}, pos=[0,10,0]) then add_modifier(id='merlon', modifier={'type':'array','count':10,'offset':[2,0,0]}).", {"id": _s("Object id"), "modifier": _modifier(), "index": _i("Insert position in the stack (default: append)")}, ["id", "modifier"]),
    tool("remove_modifier", "Remove the modifier at index (default -1 = last).", {"id": _s("Object id"), "index": _i("Stack index, negative counts from the end")}, ["id"]),
    tool("set_modifier", "Change parameters of the modifier at index, e.g. set_modifier(id='merlon', index=0, params={'count': 12}).", {"id": _s("Object id"), "index": _i("Stack index"), "params": _obj("Parameters to change")}, ["id", "index", "params"]),
    # ---------------------------------------------------------------- grouping / selection
    tool("group", "Put objects (and/or sub-groups) into a group. Groups nest; pass a group id anywhere `ids` is accepted to act on all members.", {"ids": _ids("Object ids and/or existing group ids to put in the group"), "group_id": _s("snake_case group id")}, ["ids", "group_id"]),
    tool("ungroup", "Dissolve a group (members go to its parent group, if any).", {"group_id": _s("Group id")}, ["group_id"]),
    tool("select", "Find object ids matching a query. " + SELECT_LANG, {"query": _s("Query, e.g. 'tag:tower above_y:20' or 'tower_*' or 'op:subtract'")}, ["query"]),
    # ---------------------------------------------------------------- introspection
    tool("describe", "Read the scene outline (compact text, never raw JSON): every object with shape, position, material and modifiers, grouped. detail='full' adds bboxes, tags and the material table. Call it whenever you are unsure what exists.", {"ids": _ids("Restrict to these objects (default: all)"), "detail": _s("outline (default) or full", enum=["outline", "full"])}),
    tool("bbox", "World bbox, size and centre of objects (default: the whole build). Bbox hi is exclusive: a 14-wide box at x=17 reports 10..24 and fills voxels 10..23.", {"ids": _ids("Objects (default all add-objects)")}),
    tool("measure", "Gap or overlap per axis between two objects' bboxes, and their centre distance.", {"id_a": _s("First object"), "id_b": _s("Second object")}, ["id_a", "id_b"]),
    tool("top_of", "The y of an object's top face and its centre, for stacking things on it.", {"id": _s("Object id")}, ["id"]),
    tool("side_of", "The coordinate of an object's face in a direction (north = -z side, south = +z, east = +x, west = -x, up, down) and the face centre, for attaching things to it.", {"id": _s("Object id"), "dir": _s("north, south, east, west, up or down", enum=["north", "south", "east", "west", "up", "down"])}, ["id", "dir"]),
    # ---------------------------------------------------------------- materials
    tool("define_material", "Define or update a named material. " + MATERIAL_SPEC + " Example: define_material(name='stone_wall', spec={'base':'stone_bricks','palette':[['stone_bricks',0.7],['cracked_stone_bricks',0.2],['mossy_stone_bricks',0.1]],'gradient':{'axis':'y','from':0,'to':4,'palette':[['cobblestone',0.7],['mossy_cobblestone',0.3]]},'fit':'stairs+slab'})", {"name": _s("snake_case material name"), "spec": _obj("Material spec")}, ["name", "spec"]),
    tool("list_materials", "List the materials defined in the scene and how many objects use each.", {}),
    # ---------------------------------------------------------------- history
    tool("undo", "Undo the last n scene ops (scene only; use undo_world for the game world).", {"n": _i("How many ops (default 1)")}),
    tool("redo", "Redo n undone scene ops.", {"n": _i("How many ops (default 1)")}),
    tool("snapshot", "Save the current scene under a label so you can restore it later (e.g. before an experiment).", {"label": _s("Snapshot label")}, ["label"]),
    tool("restore", "Restore a labelled snapshot (the current scene goes on the undo stack).", {"label": _s("Snapshot label")}, ["label"]),
    # ---------------------------------------------------------------- agent tools
    tool(
        "run_script",
        "THE main way to build: one script per stage that creates all of the stage's geometry/materials (loops for towers, colonnades, merlon rings). Individual op calls are capped at 12 per stage. The script gets `scene` with every scene op as a method: scene.add(id=..., shape=..., pos=...), scene.move(ids, delta), scene.define_material(name, spec), ... plus `math`, `random`, `json`, `range`, `print`. No imports, no file/network I/O, 10 s limit. Every op it performs is recorded individually (undo works). Example:\n"
        "for i, (x, z) in enumerate([(0,0),(30,0),(0,30),(30,30)]):\n    scene.add(id=f'tower_{i}', shape={'type':'cylinder','radius':4,'height':20}, pos=[x,0,z], material='stone_wall', modifiers=[{'type':'shell','thickness':1}])\n    scene.add(id=f'tower_{i}_roof', shape={'type':'cone','radius':5,'height':7}, pos=[x,20,z], material='slate_roof')",
        {"python": _s("Python source code")},
        ["python"],
    ),
    tool(
        "render",
        "Rasterise the scene and render it to images you can SEE (vision). Default: a contact sheet (iso + front + top). views: iso (45°/35°, default), front (from the south looking north — the player's view), back, left (from the west), right (from the east), top, or 'contact'. cutaway={'axis':'z','at':20} removes everything with z>20 for interiors; slice={'y':5} gives an ASCII plan at that layer. Also returns a text summary: bbox, block count, top block ids, fit counts.",
        {"views": {"type": "array", "items": {"type": "string", "enum": ["contact", "iso", "front", "back", "left", "right", "top"]}, "description": "Views to render (default ['contact'])"}, "cutaway": _obj("Optional cutaway {'axis': 'x'|'y'|'z', 'at': number}: hides voxels beyond `at` on that axis", {"axis": _s("x, y or z", enum=["x", "y", "z"]), "at": _n("Coordinate")}), "slice_y": _i("Optional: also return an ASCII plan slice at this y layer")},
    ),
    tool("lint", "Run the design-rule linter on the current scene (blank façades, roof overhang/slope, entrances, material variety, floating/isolated blocks, props on air, units mistakes). Findings cite rule numbers and object ids.", {}),
    tool("place", "Build the scene in the game world. mode='diff' (default) sends only blocks that changed since the last placement so edits feel live; 'full' resends everything. animate=true places bottom-up in timed chunks. The first placement picks the anchor 2 blocks in front of the player, front (+z) facing them, and snapshots the world for undo_world.", {"mode": _s("diff (default) or full", enum=["diff", "full"]), "animate": _b("Animate bottom-up (default true)")}),
    tool("undo_world", "Restore the game world to how it was before this session's first placement (does not touch the scene document).", {}),
    tool("export_schematic", "Export the current build as a .litematic file (Litematica) for survival players. Returns the file path.", {"name": _s("File name without extension (snake_case)")}, ["name"]),
    tool("materials_list", "Count blocks by type for the current build (a shopping list for survival: stacks and totals).", {"limit": {"type": "integer", "description": "max block types to list (default 40)"}}),
    tool("get_player", "Player position, facing (north/east/south/west), yaw/pitch and the block they are looking at.", {}),
    tool("say", "Send one short line to the player's chat for a decision they should know about. Progress is reported by the runtime — do not narrate stages.", {"text": _s("Message text (one line, <= 200 chars, no JSON)")}, ["text"]),
    tool("search_blocks", "Search the block registry by keywords (e.g. 'copper stairs', 'glazed terracotta', 'wall lantern'). Returns matching block ids with their properties. Use it before guessing an id.", {"query": _s("Keywords"), "limit": _i("Max results (default 20)")}, ["query"]),
    tool("nearest_block", "Find the full block whose average texture colour is closest to an RGB colour, optionally within a category (wool, concrete, terracotta, glass, stone, wood, ...).", {"rgb": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "[r, g, b] 0-255"}, "category": _s("Optional category filter")}, ["rgb"]),
    tool("set_brief", "Store the design brief (Interpret stage output): build_type, style, footprint [w, d], height, features, constraints, silhouette_plan (a short text plan of primary masses), palette_intent.", {"brief": _obj("Brief JSON object")}, ["brief"]),
    tool("finish", "Signal that the current stage/task is complete, with a one-line summary of what was done (shown to the player).", {"summary": _s("One line")}, ["summary"]),
]

TOOLS_BY_NAME: Dict[str, Dict[str, Any]] = {t["function"]["name"]: t for t in TOOL_SCHEMAS}

AGENT_TOOL_NAMES = ("run_script", "render", "lint", "place", "undo_world", "export_schematic", "materials_list", "get_player", "say", "search_blocks", "nearest_block", "set_brief", "finish")
HISTORY_TOOL_NAMES = ("undo", "redo", "snapshot", "restore")


def scene_op_names() -> List[str]:
    """Names of tools that are scene ops (dispatched through Session.apply)."""
    return [n for n in TOOLS_BY_NAME if n not in AGENT_TOOL_NAMES and n not in HISTORY_TOOL_NAMES]


def tool_subset(names: Sequence[str]) -> List[Dict[str, Any]]:
    """Schemas for the given tool names, in the given order (unknown names raise KeyError)."""
    return [TOOLS_BY_NAME[n] for n in names]


def all_tool_names() -> List[str]:
    return list(TOOLS_BY_NAME)
