"""The BuildProgram vocabulary: the contract between the LLM and the engine.

Everything the LLM can express here, the engine must be able to render.
Keep this strict-JSON-schema friendly: no tuples, no dicts, no unions of
objects. Defaults exist so exemplars and repairs can omit fields; the LLM is
asked to emit every field.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Shape(str, Enum):
    rect = "rect"
    circle = "circle"
    ngon = "ngon"
    cross = "cross"
    ring = "ring"


class Side(str, Enum):
    north = "north"
    east = "east"
    south = "south"
    west = "west"
    top = "top"      # stacked on the parent; align/offset position it within the parent's footprint


class Align(str, Enum):
    start = "start"
    center = "center"
    end = "end"


class RoofType(str, Enum):
    gable = "gable"
    hip = "hip"
    flat = "flat"
    parapet = "parapet"
    shed = "shed"
    mansard = "mansard"
    gambrel = "gambrel"
    cone = "cone"
    dome = "dome"
    spire = "spire"
    pagoda = "pagoda"
    none = "none"


class Framing(str, Enum):
    none = "none"
    corners = "corners"
    bays = "bays"
    tudor = "tudor"


class WindowStyle(str, Enum):
    plain = "plain"
    arched = "arched"
    tall = "tall"
    slit = "slit"
    stair_slit = "stair_slit"   # two stairs meeting at the seam: a half-wide arrow slit
    boarded = "boarded"         # opening closed with wooden trapdoors flush in the wall (shuttered, abandoned)
    gate = "gate"               # fence gates set in the wall (barns, stables, rustic)
    bars = "bars"               # iron bars in the opening (castles, dungeons, prisons, cellars, industrial)
    fence = "fence"             # wooden fence in the opening (barns, stables, open sheds, rustic)
    wall = "wall"
    round = "round"


class AttachmentKind(str, Enum):
    tower = "tower"
    chimney = "chimney"
    dormer = "dormer"
    balcony = "balcony"
    cupola = "cupola"
    buttress = "buttress"
    bartizan = "bartizan"
    porch = "porch"
    colonnade = "colonnade"
    arch = "arch"
    gallery = "gallery"
    spire = "spire"
    flagpole = "flagpole"
    awning = "awning"
    jetty = "jetty"
    cantilever = "cantilever"


class AnchorType(str, Enum):
    corner = "corner"
    wall_bay = "wall_bay"
    roof_slope = "roof_slope"
    ridge = "ridge"
    eave = "eave"
    gable = "gable"


class Gradient(str, Enum):
    none = "none"
    vertical = "vertical"     # soft blend bottom to top with a wavering boundary (stone weathering)
    bands = "bands"           # crisp horizontal bands bottom to top (lighthouse stripes, trim lines)
    horizontal = "horizontal" # soft blend west to east across the building


class Size(BaseModel):
    """Part size as a fraction of the root part's width and depth."""

    width: float = Field(description="Fraction of the root part width (root itself uses 1.0)")
    depth: float = Field(description="Fraction of the root part depth (root itself uses 1.0)")


class IntRange(BaseModel):
    min: int
    max: int


class Attach(BaseModel):
    to: str = Field(description="Name of the parent part")
    side: Side = Field(description="Which side of the parent this part touches; top stacks it on the parent's roof")
    align: Align = Field(default=Align.center, description="Where along that side it sits")
    overlap: int = Field(default=1, description="Blocks shared with the parent; larger values embed the part")
    offset: int = Field(default=0, description="Slide along the side in blocks (signed); x shift for top")
    offset_z: int = Field(default=0, description="z shift in blocks, only for side top (positive = south)")


class RoofSpec(BaseModel):
    type: RoofType
    pitch: float = Field(default=1.0, description="Blocks up per block in. 0.5 uses slabs, 1.0 stairs, 2.0 steep")
    overhang: int = Field(default=1, description="Blocks the roof extends past the wall")
    tiers: int = Field(default=1, description="Stacked roof tiers, pagoda only")
    crenellated: bool = Field(default=False, description="Merlons on the parapet, parapet only")
    ridge_axis: Literal["auto", "x", "z"] = Field(default="auto", description="Ridge direction for gable/gambrel/shed")
    profile: Literal["straight", "concave", "convex"] = Field(
        default="straight", description="Slope shape: straight; concave = gentle at the eave and steep near the top "
                                        "(East Asian, French); convex = steep at the eave and rounded at the top (bell, barrel)")
    curl: float = Field(default=0.0, description="Blocks the eave corners sweep upward, 0 to 3 (East Asian roofs use 1.5 to 2.5)")
    ridge_offset: float = Field(default=0.0, description="Gable only: shifts the ridge toward one eave, -0.4 to 0.4, giving two different slopes (saltbox)")
    edge_width: int = Field(default=0, description="Rows in from the roof boundary built in the roof_edge palette (0 = none; 1 is usual)")
    edge_lines: Literal["none", "ridge", "hips", "bands"] = Field(
        default="none", description="Also draw lines across the field in the roof_edge palette: the ridge, the ridge plus hip lines, or horizontal bands")
    band_spacing: int = Field(default=3, description="Rows between bands when edge_lines is bands")


class PartSpec(BaseModel):
    name: str
    shape: Shape = Shape.rect
    sides: int = Field(default=8, description="Polygon sides, ngon only")
    size: Size
    floors: int = 2
    floor_height: int = Field(default=4, description="Blocks per storey including the floor block")
    taper: float = Field(default=0.0, description="Fraction of width lost per floor (0 = straight walls)")
    wall_thickness: int = Field(default=1, description="Always 1 unless the player explicitly asks for thick walls; then 2")
    odd_dims: bool = Field(default=True, description="Force odd width/depth so there is a center block")
    roof: RoofSpec
    attach: Attach | None = Field(default=None, description="None marks the root part; exactly one root")
    attic: bool = Field(default=True, description="Use the roof space as storeys when the roof is tall enough: floors, stairs, "
                                                  "gable windows and walk-in dormers. false leaves it open to the rafters (vaulted halls, naves, barns)")
    role_hint: str = Field(default="", description="Free text, e.g. nave, keep, wing")


class FacadeRules(BaseModel):
    bay_width: IntRange = IntRange(min=3, max=5)
    window: WindowStyle = WindowStyle.plain
    window_width: int = 1
    window_height: int = 2
    shutters: float = Field(default=0.0, description="Probability a window gets trapdoor shutters")
    sills: bool = True
    framing: Framing = Framing.corners
    symmetry: bool = True
    max_flat_run: int = Field(default=7, description="Longest wall run without a break")
    ground_floor_taller: int = Field(default=0, description="Extra blocks of height on the ground floor")
    window_trim: Literal["none", "lintel", "surround"] = Field(
        default="none", description="none = plain openings with a sill (most buildings); lintel = one accent or trim block over each "
                                    "window (brick, plaster); surround = trim jambs and lintel (grand stone-trimmed facades only)")
    window_boxes: float = Field(default=0.0, description="0..1 chance a window gets a flowering box under its sill (cottages, inns)")
    lamp_posts: bool = Field(default=True, description="A pair of lantern posts flanking the approach to the door")
    entrance: Literal["auto", "door", "double", "portal", "gate"] = Field(
        default="auto", description="door = single door; double = pair of doors; portal = double doors with pillars, "
                                    "lintel, arch and steps; gate = 3 wide open arch with iron bars (castles). auto picks by size")


class DepthRules(BaseModel):
    frame_protrude: int = 1
    window_inset: int = 1
    eave_trim: bool = True
    foundation_rise: int = 1
    foundation_outset: int = 1
    floor_lips: bool = Field(default=False, description="Slab lips at storey boundaries, modern look")
    corbels: bool = True
    detail: float = Field(default=0.4, description="0..1 density of wall texture: stairs and wall blocks set into the wall, buttons, flush trapdoors")
    foliage: float = Field(default=0.3, description="0..1 density of vines on walls and leaf bushes at the base")
    string_courses: bool = Field(default=True, description="A notched band of stairs along each upper floor line")
    quoins: bool = Field(default=True, description="Toothed stair pattern at unframed corners")
    shading: float = Field(default=0.6, description="0..1 how strongly wall blocks darken under eaves, balconies and jetties")


class AttachmentParams(BaseModel):
    width: int | None = None
    depth: int | None = None
    height: int | None = None
    radius: int | None = None
    floors: int | None = None
    roof: RoofType | None = None
    family: str | None = Field(default=None, description="Block family override for this attachment")
    glazed: bool | None = Field(default=None, description="gallery: glaze the storey above it as a lantern room")


class AttachmentRequest(BaseModel):
    kind: AttachmentKind
    count: int | None = Field(default=None, description="How many; None lets the engine decide")
    on: str | None = Field(default=None, description="Part name, or None for anywhere")
    prefer: list[AnchorType] = []
    spacing: Literal["regular", "irregular"] = "irregular"
    params: AttachmentParams = AttachmentParams()


class SilhouetteBudget(BaseModel):
    """What the engine adds on its own when the program's attachments leave a class empty.

    dominant: elements rising above the roof (chimney, tower, cupola, spire, flagpole).
    medium: elements that interrupt roof or wall (dormer, balcony, porch, bartizan, gallery).
    small: surface details (awning, buttress, arch). Set the minimums to 0 to disable.
    """

    dominant: int = 1
    medium_min: int = 1
    medium_max: int = 3
    small_min: int = 0
    small_max: int = 2


class InteriorRules(BaseModel):
    stairs: bool = Field(default=True, description="Staircases between floors inside multi-storey parts")
    doorways: bool = Field(default=True, description="Openings in walls shared between parts")
    partitions: bool = Field(default=False, description="One partition wall in large rooms")
    lighting: bool = Field(default=True, description="Lanterns under ceilings")


class FamilyWeight(BaseModel):
    family: str = Field(description="Block family name from the catalog")
    weight: float = 1.0


class RolePalette(BaseModel):
    families: list[FamilyWeight]
    gradient: Gradient = Field(default=Gradient.none, description="vertical = soft blend of families bottom to top; bands = crisp stripes bottom to top")
    weathering: float = Field(default=0.3, description="0..1, mossy/cracked variants near the ground")
    texture_rate: float = Field(default=0.2, description="0..1, chance of a same-tone variant block")


class PaletteSpec(BaseModel):
    primary: RolePalette
    roof: RolePalette
    roof_edge: RolePalette | None = Field(default=None, description="Blockset for roof edges and dividing lines, when the roof's edge_width or edge_lines is set")
    secondary: RolePalette | None = None
    accent: RolePalette | None = None
    framing: RolePalette | None = None
    trim: RolePalette | None = None
    foundation: RolePalette | None = None
    glass: RolePalette | None = None


class Bounds(BaseModel):
    width: int
    height: int
    depth: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.width, self.height, self.depth)


class BuildProgram(BaseModel):
    label: str = Field(description="Free text name of the building type, e.g. lighthouse")
    bounds: Bounds = Field(description="Suggested bounding box; the player's selection overrides it")
    parts: list[PartSpec]
    facade: FacadeRules = FacadeRules()
    depth: DepthRules = DepthRules()
    attachments: list[AttachmentRequest] = []
    budget: SilhouetteBudget = SilhouetteBudget()
    interior: InteriorRules = InteriorRules()
    material_reasoning: str = Field(
        default="", description="One or two sentences: what the real building is made of (walls, roof, base, trim) and "
                                "which catalog family stands in for each, chosen by material, colour and texture")
    palette_name: str = Field(default="", description="A curated palette name to start from, or empty to compose freely")
    palette: PaletteSpec
    notes: str = Field(default="", description="Anything requested that could not be expressed")

    def root(self) -> PartSpec:
        for p in self.parts:
            if p.attach is None:
                return p
        return self.parts[0]
