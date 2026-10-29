add_executable(SpudGPUMeshShaders main.c)
target_link_libraries(SpudGPUMeshShaders PRIVATE spudlib SDL3::SDL3)

# Shaders are SPIR-V regardless of GRAPHICS_BACKEND -- see
# HelloConstBuffers/CMakeLists.txt for why glslc is the one shader toolchain
# every backend needs. Mesh shaders (GL_EXT_mesh_shader) need SPIR-V 1.4+,
# which glslc only emits when explicitly targeting Vulkan 1.2+ -- its default
# target (Vulkan 1.0/SPIR-V 1.0) rejects the extension outright.
find_package(Vulkan REQUIRED)
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
endif()

set(SPUDGPUMESHSHADERS_SHADER_OUTDIR "$<TARGET_FILE_DIR:SpudGPUMeshShaders>/shaders")

function(spudgpumeshshaders_compile_shader SHADER)
    get_filename_component(NAME ${SHADER} NAME)
    set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
    add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPUDGPUMESHSHADERS_SHADER_OUTDIR}"
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPV} "${SPUDGPUMESHSHADERS_SHADER_OUTDIR}/${NAME}.spv"
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
    )
    target_sources(SpudGPUMeshShaders PRIVATE ${SPV})
endfunction()

spudgpumeshshaders_compile_shader(shaders/mesh.mesh)
spudgpumeshshaders_compile_shader(shaders/scene.frag)

# Dragon_LOD0.bin (the meshlet binary, copied from d3d12samples) is loaded
# via a plain relative path at runtime, same as the shaders above.
add_custom_command(TARGET SpudGPUMeshShaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/Dragon_LOD0.bin" "$<TARGET_FILE_DIR:SpudGPUMeshShaders>")

if(WIN32)
    add_custom_command(TARGET SpudGPUMeshShaders POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>" "$<TARGET_FILE_DIR:SpudGPUMeshShaders>")
endif()
