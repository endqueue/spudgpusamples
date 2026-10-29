add_executable(SpudGPUExecuteIndirect main.c)
target_link_libraries(SpudGPUExecuteIndirect PRIVATE spudlib SDL3::SDL3)

# Shaders are SPIR-V regardless of GRAPHICS_BACKEND -- the D3D12 backend
# cross-compiles SPIR-V to HLSL at runtime via SPIRV-Cross, and the Metal
# backend cross-compiles it to MSL, both via spudlib's own shader-module
# creation (see spudlib/CLAUDE.md), so glslc is the one shader toolchain
# every backend needs.
find_package(Vulkan REQUIRED)
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
endif()

set(SPUDGPUEXECUTEINDIRECT_SHADER_OUTDIR "$<TARGET_FILE_DIR:SpudGPUExecuteIndirect>/shaders")

function(spudgpuexecuteindirect_compile_shader SHADER)
    get_filename_component(NAME ${SHADER} NAME)
    set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
    add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPUDGPUEXECUTEINDIRECT_SHADER_OUTDIR}"
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPV} "${SPUDGPUEXECUTEINDIRECT_SHADER_OUTDIR}/${NAME}.spv"
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
    )
    target_sources(SpudGPUExecuteIndirect PRIVATE ${SPV})
endfunction()

spudgpuexecuteindirect_compile_shader(shaders/scene.vert)
spudgpuexecuteindirect_compile_shader(shaders/scene.frag)
spudgpuexecuteindirect_compile_shader(shaders/cull.comp)

if(WIN32)
    add_custom_command(TARGET SpudGPUExecuteIndirect POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>" "$<TARGET_FILE_DIR:SpudGPUExecuteIndirect>")
endif()
