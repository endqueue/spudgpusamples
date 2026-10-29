add_executable(SpudGPUDepthBoundsTest main.c)
target_link_libraries(SpudGPUDepthBoundsTest PRIVATE spudlib SDL3::SDL3)

# Shaders are SPIR-V regardless of GRAPHICS_BACKEND -- see
# HelloConstBuffers/CMakeLists.txt for why glslc is the one shader toolchain
# every backend needs, Metal included (Metal cross-compiles the same SPIR-V
# to MSL, see spudlib/CLAUDE.md). This sample builds on all three backends
# (see spudgpusamples/CMakeLists.txt) -- whether spudgpu_cmd_set_depth_bounds
# actually does anything on a given Metal device depends on
# spudgpu_depth_bounds_capabilities::supported at runtime (gated to
# MTLGPUFamilyApple10, macOS/iOS 26+), not on GRAPHICS_BACKEND.
find_package(Vulkan REQUIRED)
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
endif()

set(SPUDGPUDEPTHBOUNDSTEST_SHADER_OUTDIR "$<TARGET_FILE_DIR:SpudGPUDepthBoundsTest>/shaders")

function(spudgpudepthboundstest_compile_shader SHADER)
    get_filename_component(NAME ${SHADER} NAME)
    set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
    add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPUDGPUDEPTHBOUNDSTEST_SHADER_OUTDIR}"
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPV} "${SPUDGPUDEPTHBOUNDSTEST_SHADER_OUTDIR}/${NAME}.spv"
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
    )
    target_sources(SpudGPUDepthBoundsTest PRIVATE ${SPV})
endfunction()

spudgpudepthboundstest_compile_shader(shaders/triangle.vert)
spudgpudepthboundstest_compile_shader(shaders/triangle.frag)
spudgpudepthboundstest_compile_shader(shaders/highlight.frag)

if(WIN32)
    add_custom_command(TARGET SpudGPUDepthBoundsTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>" "$<TARGET_FILE_DIR:SpudGPUDepthBoundsTest>")
endif()
