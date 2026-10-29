add_executable(SpudGPUBundles main.c)
target_link_libraries(SpudGPUBundles PRIVATE spudlib SDL3::SDL3)

# Shaders are SPIR-V regardless of GRAPHICS_BACKEND -- see
# HelloConstBuffers/CMakeLists.txt for why glslc is the one shader toolchain
# both backends need. This sample only ever builds against Vulkan/D3D12
# (see spudgpusamples/CMakeLists.txt's GRAPHICS_BACKEND guard -- Metal has
# no CPU-side reusable secondary-command/bundle mechanism, see spudlib/
# CLAUDE.md's SPUDGPU_EXT_BUNDLES section).
find_package(Vulkan REQUIRED)
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
endif()

set(SPUDGPUBUNDLES_SHADER_OUTDIR "$<TARGET_FILE_DIR:SpudGPUBundles>/shaders")

function(spudgpubundles_compile_shader SHADER)
    get_filename_component(NAME ${SHADER} NAME)
    set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
    add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPUDGPUBUNDLES_SHADER_OUTDIR}"
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPV} "${SPUDGPUBUNDLES_SHADER_OUTDIR}/${NAME}.spv"
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
    )
    target_sources(SpudGPUBundles PRIVATE ${SPV})
endfunction()

spudgpubundles_compile_shader(shaders/scene.vert)
spudgpubundles_compile_shader(shaders/scene.frag)

# occcity.bin (mesh data only here -- see main.c for why this sample skips
# the diffuse texture), copied verbatim from SpudGPUDynamicIndexing, which
# in turn copied it from d3d12samples. Loaded via a plain relative path at
# runtime, same as the shaders above.
add_custom_command(TARGET SpudGPUBundles POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/occcity.bin" "$<TARGET_FILE_DIR:SpudGPUBundles>")

if(WIN32)
    add_custom_command(TARGET SpudGPUBundles POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>" "$<TARGET_FILE_DIR:SpudGPUBundles>")
endif()
