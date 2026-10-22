# SpudGPU Samples

Ports of the [`microsoft/DirectX-Graphics-Samples`](https://github.com/microsoft/DirectX-Graphics-Samples)
(cloned alongside this repo as `../d3d12samples`) rebuilt against **SpudGPU**
(`spudlib`'s GPU HAL module) instead of raw D3D12/DXGI. Each sample under
`Samples/` mirrors one sample from `d3d12samples/Samples/Desktop/` as closely
as its feature set allows, replacing every `ID3D12*`/DXGI call with the
equivalent `spudgpu_*` call.

The point isn't a demo reel — it's a maturity test for SpudLib. Every sample
that ports cleanly validates a slice of SpudGPU's API; every sample that
*can't* port (no raytracing pipeline, no VRS, no cross-adapter API as of this
writing) is a concrete, prioritized gap to close in `spudlib` itself — not a
workaround to invent here.

## Status

| Sample | D3D12 source | Status |
|---|---|---|
| HelloTriangle | `D3D12HelloWorld/HelloTriangle` | Ported |
| HelloConstBuffers | `D3D12HelloWorld/HelloConstBuffers` | Ported |
| HelloTexture | `D3D12HelloWorld/HelloTexture` | Ported, all backends (see note below) |
| SpudGPUExecuteIndirect | `D3D12ExecuteIndirect` | Ported (see note below) |
| SpudGPUDynamicIndexing | `D3D12DynamicIndexing` | Ported, Vulkan/D3D12 only (see note below) |
| SpudGPUMeshShaders | `D3D12MeshShaders/MeshletRender` | Ported, all backends (see note below) |
| SpudGPUBundles | `D3D12Bundles` | Ported, Vulkan/D3D12 only (see note below) |
| SpudGPUDepthBoundsTest | `D3D12DepthBoundsTest` | Ported, all backends (see note below) |
| SpudGPUHeaps | `D3D12SmallResources` | Ported, all backends; run on D3D12 and Vulkan, not yet on Metal (see note below) |

`HelloTexture` closed a real gap and worked around another. The gap it
closed: `spudgpu_cmd_copy_buffer_to_image`/`_copy_image_to_buffer` and
`spudgpu_get_image_buffer_copy_size` were unimplemented placeholders on
Metal — `spudgpumetalrenderpass.m`'s file comment already flagged its
copy/blit functions as such, "no caller needs them yet" (see
`SpudGPUMeshShaders` above, whose buffer-to-buffer copy hit the same
placeholder). This sample's texture upload was the first caller to actually
need buffer-to-image, and it silently produced a solid-black texture until
these were implemented for real via `MTLBlitCommandEncoder`
(`copyFromBuffer:toTexture:`/`copyFromTexture:toBuffer:`), gated behind a new
`_active_blit_encoder` on the command list that ends any active render/compute
encoder before opening (mirroring `_active_compute_encoder`'s existing
pattern) since Metal disallows two live encoders of different kinds on one
command buffer at once.

The gap this port works around instead of closing:
`SPUDGPU_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` — declared in `spudgpu.h` and
implemented on Vulkan/D3D12 since the type was added — doesn't actually work
on Metal. `spudgpumetaldescriptors.m` already documented that a combined
descriptor's paired sampler is "intentionally not written" there, and
cross-compiling a GLSL `sampler2D` confirms why: under SpudGPU's
per-descriptor-set-argument-buffer scheme, SPIRV-Cross's synthesized sampler
member overlaps the texture's own binding slot, which it only permits via
"full mutable aliasing" on Metal 3+ (`spirv_msl.cpp`). This port doesn't fix
that gap — it works around it the same way `SpudGPUDynamicIndexing`'s
bindless design already had to: a `texture2D` at one binding and a plain
`sampler` at the next, written as separate `SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_
IMAGE`/`SPUDGPU_DESCRIPTOR_TYPE_SAMPLER` descriptors instead of one combined
one. This is the first sample to exercise that split-descriptor idiom outside
bindless, and the first to exercise ordinary (non-bindless) texture sampling
on Metal at all. Two smaller, harmless deviations alongside it: a real
dynamic `spudgpu_sampler` stands in for the original's static/immutable
root-signature sampler (`spudgpu_sampler_desc` has no such mechanism yet —
see `spudlib/CLAUDE.md`'s "Known gaps"), and `CLAMP_TO_EDGE` addressing
replaces the original's `CLAMP_TO_BORDER` with a transparent-black border
(`spudgpu_sampler_desc` has no border-color field) — this sample's UVs never
leave `[0, 1]`, so the two modes look identical here regardless.

`SpudGPUExecuteIndirect` closed a real gap: `spudgpu` had no compute-dispatch
call at all, and no buffer pipeline barriers on any backend, both now added
alongside `spudgpu_cmd_draw_indirect`/`_indexed_indirect`. The port deviates
from the original in one deliberate, documented way, and initially shipped a
second deviation that turned out to be a genuine D3D12 portability bug rather
than a valid simplification (see below):

- **No GPU-side compaction.** The original appends visible triangles into a
  tightly-packed buffer (an HLSL `AppendStructuredBuffer`, i.e. a UAV atomic
  counter) and issues a GPU-authored *variable* draw count
  (`ExecuteIndirect`'s count-buffer argument / `vkCmdDrawIndirectCount`).
  Metal has no equivalent primitive for that shape — a GPU-authored variable
  indirect draw count needs an `MTLIndirectCommandBuffer`, a structurally
  different object encoded via its own API. This port's compute pass instead
  writes one draw-args entry per triangle at a fixed index (zeroed
  `vertex_count` for a culled triangle), and the renderer always issues a
  single fixed-count `spudgpu_cmd_draw_indirect` — fully portable, same
  demonstrated mechanism (a compute pass decides per-triangle what gets
  drawn), just without the bandwidth-compaction optimization.
- **No per-draw root CBV update, and no `gl_InstanceIndex` either.** The
  original's command signature updates a root constant-buffer-view descriptor
  per draw, alongside the draw itself — a D3D12-specific indirect-argument
  shape with no Vulkan/Metal equivalent, so this port doesn't use it. The
  first attempt at a portable substitute baked each triangle's index into its
  draw's `first_instance` and read it back in the vertex shader via
  `gl_InstanceIndex`/`SV_InstanceID` — this is broken on D3D12: unlike
  Vulkan's `gl_InstanceIndex`, D3D12's `SV_InstanceID` does not include
  `StartInstanceLocation`, so every one of the 256 indirect draws read
  instance 0's data regardless of its actual `first_instance` (SPIRV-Cross
  can emulate this via a `SPIRV_Cross_BaseInstance` root constant, but that
  constant has to be fed per sub-draw through `ExecuteIndirect`'s own
  per-draw root-constant-update argument — exactly the D3D12-only mechanism
  this port was trying to avoid). The actual portable substitute: `scene`'s
  per-triangle offset/color is bound as a second, **per-instance vertex
  buffer** (`per_instance = true` on its `spudgpu_vertex_binding_desc`)
  instead of a storage/uniform buffer indexed by instance ID. The
  fixed-function input assembler's per-instance vertex fetch *does* correctly
  offset by `StartInstanceLocation` on every backend — this is a separate
  mechanism from the `SV_InstanceID`/`gl_InstanceIndex` system value, and is
  the standard, actually-portable technique for per-instance data. A
  `first_instance` other than 0 in an indirect draw's arguments needs a
  device created with `SPUDGPU_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE`
  (Vulkan's `drawIndirectFirstInstance`; every D3D12 and Metal adapter), so
  the sample asks for it and stops on an adapter without it.

`SpudGPUDynamicIndexing` exercises `spudgpu`'s bindless descriptor indexing
feature for the first time (`spudgpu_bindless_register_sampled_image`,
`spudgpu_cmd_bind_bindless_resources`, ...) — a 15x8 grid of building meshes,
each dynamically indexing into one shared unbounded texture table for its own
procedurally-generated material color. That feature was already fully
implemented on Vulkan/D3D12 before this port (just never exercised by a
sample); the one real gap this port closed was **SpudGPU had no sampler
object on any backend** — now added (`spudgpu_create_sampler`). Bindless is
`0` on Metal (needs a `MTLHeap`-backed allocator SpudGPU doesn't have yet —
see `spudlib/CLAUDE.md`), so this sample only builds against Vulkan/D3D12;
`spudgpusamples/CMakeLists.txt` skips it entirely when
`GRAPHICS_BACKEND=Metal`. One deliberate simplification: the original's
diffuse texture is `BC1_UNORM` (block-compressed); this port decodes it to
plain `R8G8B8A8_UNORM` on the CPU at load time rather than exercising
`spudgpu`'s (real, but never-yet-uploaded-to) BC1 format support, to avoid
stacking an unverified compressed-texture-upload path on top of bindless
textures, a real sampler, and a real depth buffer all being exercised for
the first time in one sample.

`SpudGPUMeshShaders` closed the biggest remaining gap this suite tracked:
`spudgpu` had no mesh-shader pipeline at all (mesh shaders replace
vertex-fetch/input-assembly entirely with a shader stage that emits meshlet
geometry directly) — now added on **all three** backends (`VK_EXT_mesh_shader`
on Vulkan, a hand-rolled `D3D12_PIPELINE_STATE_STREAM_DESC` on D3D12 since the
vendored `d3dx12.h` predates Microsoft's own mesh-PSO helpers, and
`MTLMeshRenderPipelineDescriptor` on Metal), gated behind the new
`SPUDGPU_EXT_MESH_SHADING` — see `spudlib/CLAUDE.md` for why this `EXT` is a
different flavor from bindless's (the macro is `1` everywhere; it's *runtime*
driver/hardware support that varies, surfaced via
`spudgpu_get_mesh_shading_capabilities`). The sample renders the real
`Dragon_LOD0.bin` meshlet asset shipped with the original D3D12 sample,
one GLSL mesh shader (`GL_EXT_mesh_shader`) cross-compiled to all three
targets exactly like every other sample's shaders. This sample's four
meshlet data buffers need `SPUDGPU_BUFFER_USAGE_STORAGE`, which
`SPUDGPU_MEMORY_KIND_UPLOAD` refuses on every backend — the rule comes from
D3D12, where an UPLOAD-heap resource can't carry
`D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`, and SpudGPU holds every
backend to it (the mappable kind that does take `STORAGE`,
`SPUDGPU_MEMORY_KIND_DEVICE_MAPPABLE`, exists only on some devices).
Getting this sample running on D3D12 closed a real gap in `spudgpu` itself:
**`spudgpu_cmd_copy_buffer` (buffer-to-buffer copy) didn't exist on any
backend**, so the textbook fix — stage the CPU data through a small
`TRANSFER_SRC` buffer, then copy into a real `SPUDGPU_MEMORY_KIND_DEVICE` buffer — wasn't
available. Now added (`vkCmdCopyBuffer` / `CopyBufferRegion`; Metal's
`MTLBlitCommandEncoder` copy was initially left as an unimplemented
placeholder like this file's other blit functions, unverified without Apple
hardware — see `HelloTexture` below for where that caught up with this
sample), each of the four buffers gets a small synchronous staging upload
plus a `COMMON` →
`UNORDERED_ACCESS` pipeline barrier (needed only on D3D12 in practice — that
transition isn't one of the states D3D12 promotes to implicitly the way it
does `COPY_DEST`/read-only states, but costs nothing on Vulkan given the
`spudgpu_queue_wait_idle` already between the copy and first use).

`SpudGPUBundles` closed the last real gap `SPUDGPU_COMMAND_LIST_TYPE_BUNDLE`
had been sitting on since it was added to the base command-list-type enum:
the type existed and D3D12's allocator/list creation already honored it, but
nothing could actually *execute* a bundle once recorded. Now added, gated
behind the new `SPUDGPU_EXT_BUNDLES` (`spudgpu_begin_bundle_command_list`,
`spudgpu_cmd_execute_bundle`) — `1` on Vulkan/D3D12, `0` on Metal, the same
flavor of `EXT` as bindless (a structural capability gap, not a runtime
driver check): Metal's `MTLCommandBuffer`/`MTLRenderCommandEncoder` are
single-use, so there's no CPU-side reusable secondary-command mechanism to
record a bundle into at all. D3D12's implementation is the straightforward
native case (`ID3D12GraphicsCommandList::ExecuteBundle`); Vulkan's bundle is
a `VK_COMMAND_BUFFER_LEVEL_SECONDARY` buffer replayed via
`vkCmdExecuteCommands`, which — since this codebase is dynamic-rendering-only
(no `VkRenderPass`) — needed `VkCommandBufferInheritanceRenderingInfo` wired
through a new `spudgpu_begin_bundle_command_list`/`spudgpu_bundle_inheritance_
desc` (the attachment formats a secondary buffer will replay under, since
there's no framebuffer object to infer them from) plus a new
`will_execute_bundles` flag on `spudgpu_rendering_begin_desc` so the primary
list's `vkCmdBeginRendering` opens with
`VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT` only when the caller
actually intends to execute one — SpudLib never infers that on the caller's
behalf (see `spudlib/CLAUDE.md`'s zero-policy rule). The sample itself
demonstrates the mechanism the original does: a `CityRowCount x
CityColumnCount` (10x3) grid of `occcity` buildings (the same asset
`SpudGPUDynamicIndexing` uses), each with its own per-object CBV, drawn
either by replaying one bundle recorded once at startup (default) or by
re-recording the identical bind+draw sequence into the direct command list
every frame — press `C` to toggle, matching the original's own keybinding.
Two deliberate simplifications versus the original: no diffuse texture (this
sample isn't about texturing — each building instead gets a fixed
HSL-gradient color baked into its CBV, so it only exercises the bundle
mechanism itself), and single-buffered CBV storage overwritten in place every
frame, matching every other sample in this suite, rather than the original's
3 rotating `FrameResource`s — the bundle's recorded descriptor-set bindings
point at fixed buffer offsets regardless, so this doesn't touch the actual
mechanism being demonstrated.

`SpudGPUDepthBoundsTest` closed a gap that had sat unaddressed since the very
first sample: `spudgpu` had no depth bounds test at all — a mechanism, distinct
from the ordinary per-fragment depth test, that discards a fragment based on
whether the depth attachment's *existing* value at that pixel (already
written by an earlier draw) falls inside a caller-set `[min, max]` window, set
per-draw via the new `spudgpu_cmd_set_depth_bounds`. Added on Vulkan
(`VkPhysicalDeviceFeatures::depthBounds` + `vkCmdSetDepthBounds`, both core
1.0, no extension), D3D12 (`D3D12_DEPTH_STENCIL_DESC1::
DepthBoundsTestEnable` + `ID3D12GraphicsCommandList1::OMSetDepthBounds`), and
Metal (`[MTLRenderCommandEncoder setDepthTestMinBound:maxBound:]`, macOS/iOS
26+), gated behind the new `SPUDGPU_EXT_DEPTH_BOUNDS_TEST` — `1` on **every**
backend, since unlike bindless/bundles this isn't a structural gap on Metal
(the primitive genuinely exists there, just very recently added). What does
vary per-device is runtime hardware support, the `SPUDGPU_EXT_MESH_SHADING`
flavor of this pattern rather than the bindless/bundles flavor:
`spudgpu_depth_bounds_capabilities::supported` reflects
`VkPhysicalDeviceFeatures::depthBounds` (Vulkan),
`D3D12_FEATURE_DATA_D3D12_OPTIONS2::DepthBoundsTestSupported` (D3D12), or
`[MTLDevice supportsFamily:MTLGPUFamilyApple10]` (Metal — the newest Apple
GPU family as of this writing, so expect `false` on most Metal hardware in
the field today even though the sample still builds and runs there). Metal
also has no pipeline-level enable toggle at all, unlike Vulkan/D3D12 —
`spudgpu_shader_pipeline_desc::depth_bounds_test_enable` is ignored there,
and `spudgpu_cmd_set_depth_bounds` alone both enables and configures the test
per draw (passing `(0, 1)` disables it, matching Metal's own default-disabled
semantics). Adding `DepthBoundsTestEnable` to D3D12's pipeline creation also
closed a smaller, adjacent gap: that field only exists on
`D3D12_DEPTH_STENCIL_DESC1`, reachable only through `CreatePipelineState`'s
PSO-stream path — `spudgpu_create_shader_pipeline`'s classic (non-mesh)
pipeline path was still going through the older `CreateGraphicsPipelineState`/
`D3D12_GRAPHICS_PIPELINE_STATE_DESC`, so it now builds a
`CD3DX12_PIPELINE_STATE_STREAM1` from that same desc instead (a lossless,
purely additive upgrade — see `spudgpud3d12shader.cpp`) rather than
maintaining two divergent pipeline-creation paths.

The port itself deviates from the original in one deliberate, documented way:
**no colorless depth-only priming pass.** The D3D12 original's first draw
uses a PSO with no pixel shader at all (`DEPTH_ONLY_PSO_STREAM` has no `PS`
subobject) purely to write real per-vertex depth into the depth buffer without
touching the render target's color. `spudgpu_create_shader_pipeline` requires
a fragment module on every pipeline (`SPUDRESULT_GPU_VERTEX_AND_FRAGMENT_
SHADER_REQUIRED` otherwise) — lifting that would mean threading an optional
fragment stage (and, to fully match the original's *visual* result, a
color-write-mask override neither backend's `spudgpu_blend_attachment_desc`
exposes yet) through both backends for a single sample's sake. Instead, this
port's priming pass renders the triangle's real colors and real depth (an
ordinary opaque draw), and the depth-bounds-tested second pass redraws the
same triangle in a fixed highlight color instead of the original's vertex
colors — wherever the bounds window clips a fragment, the first pass's real
color already shows through underneath, so the mechanism being demonstrated
(an animated depth-bounds window carving a band out of the triangle) reads
just as clearly, just with a highlight-over-base look rather than
background-showing-through.

`SpudGPUHeaps` is `D3D12SmallResources` under the name of the SpudGPU
feature it exercises: heaps and the resources placed in them
(`spudgpu_create_heap`, `spudgpu_get_*_placement_requirements`,
`spudgpu_create_placed_buffer`/`_image`). The grid of 77 small textures is
made either placed, one after another in a single heap, or committed, one
device allocation each; Space switches between them, as in the original.
The port goes further than the original in one place and falls short of it
in two:

- **Buffers are placed too.** The vertex buffer goes in a heap of device
  memory, and the staging buffer that fills it and the textures in a heap of
  upload memory, where `SPUDGPU_DEVICE_PROPERTIES::mappable_heaps` allows it
  (a Mac without Apple silicon can't map a heap; the staging buffer is then
  made with `spudgpu_create_buffer`).
- **No memory figure for committed resources.** The original's title shows
  `QueryVideoMemoryInfo`'s figure for each mode. SpudGPU has no memory
  budget query yet, so the title shows the placed mode's heap size and
  per-texture requirements, and says nothing measured for the committed
  mode.
- **No small-resource packing on D3D12 yet.** The original exists to show
  `D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT` (4 KiB) letting small textures
  share 64 KiB pages. SpudGPU's D3D12 backend places every texture at
  64 KiB for now (see `spudlib/CLAUDE.md`), so on D3D12 the two modes cost
  the same. On Vulkan and Metal the textures pack at whatever alignment the
  driver reports.

The sample was written against SpudGPU's heap calls the day they landed. It
type-checks against the Vulkan and Metal configurations of `spudgpu.h` and
its shaders compile with glslang. It was first built and run on 2026-10-22,
on D3D12 and Vulkan on Windows (see the end of this file); it has not run on
Metal.

See `../spudlib/CLAUDE.md` and `../CLAUDE.md` for the architecture this
repo sits alongside.

## Building

Each sample is a standalone executable target linking `spudlib`. Windowing
and surface creation use SDL3 + `spudgpu_sdl3.h` (the same approach as
`Erethal_SDL3_ImGui`, which is deprecated), so samples build and run identically on Windows and
Linux. Pick a graphics backend via the `GRAPHICS_BACKEND` cache variable
(`Vulkan` or `D3D12`), same as `spudlib` itself.

```
cmake --preset windows-vulkan   # or windows-d3d12
cmake --build build-windows-vulkan --target HelloTriangle
```

On Apple Silicon, samples also build and run against `GRAPHICS_BACKEND=Metal`
(see `spudlib/CLAUDE.md`) — except `SpudGPUDynamicIndexing`, which needs
bindless descriptor indexing, and `SpudGPUBundles`, which needs
`SPUDGPU_EXT_BUNDLES`; neither is buildable on Metal (see above).
`SpudGPUDepthBoundsTest` *does* build and run on Metal — its
`SPUDGPU_EXT_DEPTH_BOUNDS_TEST` gap isn't structural there — but
`spudgpu_depth_bounds_capabilities::supported` will be `false` on anything
short of `MTLGPUFamilyApple10` hardware, so the animated bounds window has
no visible effect on most Metal devices today even though the sample runs.

## Running

Every sample takes two optional arguments:

- `--frames N`: render N frames, then leave the render loop as closing the
  window does, through the same cleanup. Without it a sample runs until its
  window is closed. Use it to run a sample unattended, under a validation
  layer or a debugger.
- `--device N`: use the Nth device `spudgpu_enumerate_devices` lists (0 if
  not given).

## Validation run, 2026-10-16

On Windows 11, Debug, with the `windows-vulkan` preset and
`VK_LAYER_KHRONOS_validation`, `--frames 300`, on both devices of one
laptop (Intel Iris Xe, `--device 0`; NVIDIA RTX A2000, `--device 1`), every
sample exits cleanly with no validation message. `SpudGPUMeshShaders` exits 1
on the Intel device, by design: that driver reports no mesh shading.

Before that run the samples no longer compiled: SpudGPU had replaced
`color_attachment_format` with `color_attachment_formats[]` and
`color_attachment_count`. Once built, the run found:

- `SpudGPUBundles`: SpudGPU's header said a bundle inherits the direct
  list's viewport and scissor, which a Vulkan secondary command buffer does
  not; the Intel driver crashed on the bundle's first draw. A bundle is now
  given its viewport and scissor in `spudgpu_bundle_inheritance_desc`, and
  `spudgpu.h` states the bundle rules for every backend.
- `SpudGPUDynamicIndexing`: the depth image's transition was recorded into
  the upload list after it had been closed and submitted, so the depth image
  was never transitioned, and the NVIDIA driver crashed recording into the
  finished list. The transition is now recorded each frame.
- `SpudGPUExecuteIndirect`: SpudGPU's Vulkan device never enabled
  `multiDrawIndirect`, so a 256-draw indirect call was invalid. It is enabled
  where reported, and `SPUDGPU_DEVICE_PROPERTIES::max_draw_indirect_count`
  says how many draws one call takes; the sample splits its call by it.
- `SpudGPUMeshShaders`: SpudGPU's Vulkan `spudgpu_cmd_push_constants` knew
  only vertex, fragment and compute stages and dropped a mesh-stage push.
  With that fixed, the sample read freed memory every frame (its meshlet
  subsets pointed into the file buffer it had freed); they are now copied.

The same day, with the `windows-d3d12` preset (Debug) under cdb, every
sample runs 300 frames with no D3D12 debug layer message on both devices:
the Iris Xe (`--device 0`) and the NVIDIA RTX A2000 (`--device 1`).
`SpudGPUMeshShaders` exits 1 on the Iris Xe, which has no mesh shading. The
Iris Xe was listed on D3D12 only after SpudGPU's D3D12 backend stopped
requiring feature level 12_2 (it now requires 12_0); before that, D3D12
listed the NVIDIA device alone, as device 0. Two samples failed there
first, both from SpudGPU's D3D12 backend:

- `SpudGPUMeshShaders`: `CreateGraphicsPipelineState` failed, the root
  signature not covering the shader's `t3`. SPIRV-Cross declared the GLSL
  `readonly` storage buffers as SRVs, while the backend's root signature and
  descriptors make every storage buffer a UAV; it now compiles them as UAVs.
- `SpudGPUDynamicIndexing`, and `SpudGPUMeshShaders` once its pipeline was
  made: two errors on every draw, a bound table in a descriptor heap that
  was no longer set. Each descriptor pool, and the bindless state, had its
  own shader-visible heaps, and D3D12 sets one CBV/SRV/UAV heap and one
  sampler heap at a time, so binding a set from a second pool unbound the
  first's. They now all take ranges of the device's two heaps.

Later on 2026-10-16 every sample ran again, the same way, against SpudLib
with the Mac's Metal work, the D3D12 fix for 2D views of a layer past the
first and for cube views to render to or store into, storage cube images
in D3D12 shaders, and Vulkan's `imageCubeArray`. On Vulkan every sample
exited cleanly with no validation message on both devices (`SpudGPUMeshShaders`
exits 1 on the Iris Xe, as before). On D3D12 there was no debug layer
message on either device but one: `SpudGPUMeshShaders` on the Iris Xe
returned from `main` without destroying the instance when it found no mesh
shading, and the debug layer reported the device's live objects at process
exit. Every sample now leaves `main` through one cleanup on every failure:
each object `main` makes starts as NULL, a failure jumps to the cleanup at
the end, which waits for the queue if there is one and destroys whatever
exists, and the helpers that called `exit(1)` (reading a file, loading a
shader, uploading a texture, `SpudGPUHeaps`' `check`) return NULL or false
for the caller to jump. Only a failed `SDL_Init` still returns directly. Run
again on both devices and both backends, every sample is clean as above, and
each run with `--device 9` (a failure right after the instance is made)
reports no live objects on D3D12. Under cdb, each run on the RTX A2000 also logs two first-chance
C++ exceptions inside Intel's D3D12 driver as it loads, caught there, with
no debug layer message.

On 2026-10-22 every sample was built (Debug) and run again on D3D12 and
Vulkan on Windows 11, on the Intel Iris Xe and the NVIDIA RTX A2000, and on
D3D12's WARP adapter, after SpudGPU's adapter and device change. That was the
first build and run of `SpudGPUHeaps` on any backend. Neither backend built at
first: SpudAudio's WASAPI backend named `max_align_t`, which MSVC's C compiler
doesn't have (`SPUD_MAX_ALIGN`, `spudlib/src/spudallocator.h`, now). The plain
runs, 300 frames each, were clean on both backends: no validation layer or
debug layer message and exit 0, but for `SpudGPUMeshShaders` on the Iris Xe,
which has no mesh shading and exits 1 as before. Three stricter runs found two
Vulkan bugs, both fixed in SpudLib:

- Synchronization validation (`VK_KHRONOS_VALIDATION_VALIDATE_SYNC=1`)
  reported a WRITE_AFTER_READ against `vkAcquireNextImageKHR` in every
  sample: the barrier that takes a back buffer out of `UNDEFINED` or
  `PRESENT_SRC` started at `TOP_OF_PIPE` (or `BOTTOM_OF_PIPE` out of
  `SPUDGPU_RESOURCE_STATE_PRESENT`), which doesn't chain to the acquire
  semaphore's wait at `COLOR_ATTACHMENT_OUTPUT`. Such a barrier now starts at
  its destination stage.
- GPU-assisted validation (`VK_LAYER_GPUAV_ENABLE=1`) reported
  `VUID-VkDrawIndirectCommand-firstInstance-00501` in
  `SpudGPUExecuteIndirect`: its draws' `first_instance` is the triangle's
  index, and the Vulkan device never enabled `drawIndirectFirstInstance`.
  That is `SPUDGPU_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE` now, which the
  sample asks for.

D3D12 with GPU-based validation on (a temporary
`SetEnableGPUBasedValidation(TRUE)`, both devices) reported nothing. After
the fixes every sample is clean on both backends and both devices in every
mode above, and a screenshot of each sample on the RTX A2000 showed the same
image on D3D12 and on Vulkan.
