"""Free-form objects (statues, creatures, vehicles, props): text -> image -> mesh -> coloured voxels.

The building pipeline asks an LLM to compose a `BuildProgram` for a deterministic engine. Objects take a
different route because no LLM composes a convincing dragon out of primitives: an image model draws the
subject (FLUX on Azure), a local single-image reconstructor turns it into a mesh (TripoSR, in
`tools/.venv-3d`), and `voxelize` turns the mesh into a `SemanticGrid` of concrete blocks by nearest
palette colour. The LLM's only job is the small one it is good at: turning the request into an image
prompt, a height and a palette hint (`brief`).
"""
