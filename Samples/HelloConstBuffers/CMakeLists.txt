add_executable(HelloConstBuffers main.c)
target_link_libraries(HelloConstBuffers PRIVATE spudlib SDL3::SDL3)

# Shaders are SPIR-V regardless of GRAPHICS_BACKEND -- the D3D12 backend
# cross-compiles SPIR-V to HLSL at runtime via SPIRV-Cross (see
# spudlib/CLAUDE.md), so glslc is the one shader toolchain both backends need.
find_package(Vulkan REQUIRED)
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
endif()

set(HELLOCONSTBUFFERS_SHADER_OUTDIR "$<TARGET_FILE_DIR:HelloConstBuffers>/shaders")

function(helloconstbuffers_compile_shader SHADER)
    get_filename_component(NAME ${SHADER} NAME)
    set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
    add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${HELLOCONSTBUFFERS_SHADER_OUTDIR}"
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPV} "${HELLOCONSTBUFFERS_SHADER_OUTDIR}/${NAME}.spv"
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
    )
    target_sources(HelloConstBuffers PRIVATE ${SPV})
endfunction()

helloconstbuffers_compile_shader(shaders/scene.vert)
helloconstbuffers_compile_shader(shaders/scene.frag)

if(WIN32)
    add_custom_command(TARGET HelloConstBuffers POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>" "$<TARGET_FILE_DIR:HelloConstBuffers>")
endif()
