cmake_minimum_required(VERSION 3.15)
project(SpudGPUSamples LANGUAGES C CXX)

# Pull in spudlib if nobody above us already has (mirrors ApricotFields'
# CMakeLists.txt). When built through the eqdev root aggregator, spudlib is
# already a target by the time this runs and CMAKE_MSVC_RUNTIME_LIBRARY has
# already been force-set there -- see ../CLAUDE.md. When built standalone,
# set CMAKE_MSVC_RUNTIME_LIBRARY yourself before configuring to match
# whatever else you link against.
if(NOT TARGET spudlib)
    add_subdirectory(../spudlib ${CMAKE_CURRENT_BINARY_DIR}/spudlib)
endif()

# NOT ENABLED YET: a FetchContent path so this repo can be `git clone`d
# standalone by the public, without the ../spudlib sibling-checkout
# assumption above (which only holds inside the eqdev workspace via
# sync.sh). spudlib/spudgpu is still an immature, actively-changing product
# (Metal backend only just went from scaffolded to real; bindless-on-Metal
# is a known, deliberate gap) with no tagged releases -
# pinning a public sample repo's build to it now would mean either floating
# on a raw commit SHA with no compatibility guarantee, or cutting a "v0.1.0"
# that overpromises stability the library doesn't have yet. Revisit once
# spudlib has an actual first tagged release to pin to.
#
# if(NOT TARGET spudlib)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../spudlib/CMakeLists.txt")
#         # Inside the eqdev workspace - build against the live sibling
#         # checkout so in-progress spudlib edits are picked up directly.
#         add_subdirectory(../spudlib ${CMAKE_CURRENT_BINARY_DIR}/spudlib)
#     else()
#         # Standalone public clone - fetch spudlib directly.
#         include(FetchContent)
#         FetchContent_Declare(
#             spudlib
#             GIT_REPOSITORY https://github.com/vigatel2717/spudlib.git
#             GIT_TAG <pinned commit or tag>
#         )
#         FetchContent_MakeAvailable(spudlib)
#     endif()
# endif()

# SDL3 supports Windows, Linux, and macOS, so samples build on all three,
# including GRAPHICS_BACKEND=Metal on Apple Silicon (see spudlib/CLAUDE.md).
if(WIN32)
    if(NOT SDL3_DIR)
        set(SDL3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../SDL3-3.4.8/cmake")
    endif()
endif()
find_package(SDL3 REQUIRED)

add_subdirectory(Samples/HelloTriangle)
add_subdirectory(Samples/HelloConstBuffers)
add_subdirectory(Samples/HelloTexture)
add_subdirectory(Samples/SpudGPUExecuteIndirect)

# Bindless descriptor indexing (SPUDGPU_EXT_BINDLESS_DESCRIPTOR_INDEXING) is
# 0 on Metal -- it needs a MTLHeap-backed resource allocator SpudGPU doesn't
# have yet (see spudlib/CLAUDE.md). Calling any bindless function against a
# Metal build is a compile/link error by design, so this sample is only
# buildable against Vulkan/D3D12.
if(NOT GRAPHICS_BACKEND STREQUAL "Metal")
    add_subdirectory(Samples/SpudGPUDynamicIndexing)
endif()

# Mesh shading (SPUDGPU_EXT_MESH_SHADING) is implemented on all three
# backends, so this sample builds everywhere -- no guard needed.
add_subdirectory(Samples/SpudGPUMeshShaders)

# Bundles (SPUDGPU_EXT_BUNDLES) are 0 on Metal -- MTLCommandBuffer/
# MTLRenderCommandEncoder are single-use, so there's no CPU-side reusable
# secondary-command mechanism to record a bundle into at all (see
# spudlib/CLAUDE.md). Calling spudgpu_cmd_execute_bundle against a Metal
# build is a compile/link error by design, so this sample is only buildable
# against Vulkan/D3D12.
if(NOT GRAPHICS_BACKEND STREQUAL "Metal")
    add_subdirectory(Samples/SpudGPUBundles)
endif()

# Depth bounds test (SPUDGPU_EXT_DEPTH_BOUNDS_TEST) is implemented on all
# three backends -- unlike bindless/bundles, Metal's
# [MTLRenderCommandEncoder setDepthTestMinBound:maxBound:] (macOS/iOS 26+) is
# a real primitive, not a structural gap, so this sample builds everywhere.
# Whether a given device actually honors it at runtime
# (spudgpu_depth_bounds_capabilities::supported, gated to MTLGPUFamilyApple10
# on Metal) is this sample's own concern, not a build-time one -- see
# spudlib/include/spudgpu.h.
add_subdirectory(Samples/SpudGPUDepthBoundsTest)

# Heaps and placed resources have no SPUDGPU_EXT_* gate: all three backends
# implement them, so this sample builds everywhere. A device whose heaps
# can't be mapped (SPUDGPU_DEVICE_PROPERTIES::mappable_heaps false: a Mac
# without Apple silicon) still runs it, with its staging buffer made on its
# own instead of placed.
add_subdirectory(Samples/SpudGPUHeaps)
