//
// SpudGPU port of D3D12ExecuteIndirect from d3d12samples. A compute pass
// decides, per triangle, whether it's visible; the graphics pass then draws
// every triangle with one spudgpu_cmd_draw_indirect call, reading its
// per-triangle draw arguments straight out of the compute pass's output
// buffer. SPACE toggles the compute culling pass on/off -- see ../../README.md
// for the deliberate simplification versus the original D3D12 sample (flat
// indirect draw instead of GPU-side compaction) and why it's the correct
// portable choice rather than a workaround.
//
// Per-triangle offset/color is delivered as a per-instance vertex attribute
// (scene_buffer bound at vertex binding slot 1, per_instance = true), NOT via
// gl_InstanceIndex/SV_InstanceID indexing into a storage buffer. D3D12's
// SV_InstanceID does not include StartInstanceLocation the way Vulkan's
// gl_InstanceIndex does -- unlike the fixed-function input assembler's
// per-instance vertex fetch, which *does* correctly offset by
// StartInstanceLocation (first_instance) on every backend. That's the
// portable equivalent of the original sample's per-draw root CBV update
// (see ../../README.md) without needing an indirect-argument shape D3D12
// alone can do.
//

#include <SDL3/SDL.h>
#include <spudgpu.h>
#include <spudgpu_sdl3.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if SPUDLIB_PLATFORM_WINDOWS
#define _USE_MATH_DEFINES // MSVC <math.h> checks this, needed to use M_PI macro
#include <corecrt_math_defines.h>
#endif

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

// Must match the constants of the same name in shaders/cull.comp/scene.vert.
#define TRIANGLE_COUNT 256
#define COMPUTE_THREAD_BLOCK_SIZE 128
#define TRIANGLE_HALF_WIDTH 0.05f
#define TRIANGLE_DEPTH 1.0f
#define CULL_OFFSET 0.5f

typedef struct Vertex {
	float position[3];
} Vertex;

// Matches shaders/cull.comp's/scene.vert's SceneConstantBuffer block exactly
// (std140: three vec4s need no manual padding). Unlike the original D3D12
// sample, projection isn't duplicated per-triangle here (it's identical for
// every triangle there too) -- see ProjectionBuffer below -- which keeps this
// well under Vulkan's guaranteed-minimum uniform buffer size at
// TRIANGLE_COUNT triangles.
typedef struct SceneConstantBuffer {
	float velocity[4];
	float offset[4];
	float color[4];
} SceneConstantBuffer;

// Matches spudgpu_draw_indirect_args exactly (see spudgpu.h) -- the compute
// shader reads/writes this shape directly, no repacking.
typedef struct IndirectCommand {
	uint32_t vertex_count;
	uint32_t instance_count;
	uint32_t first_vertex;
	uint32_t first_instance;
} IndirectCommand;

static float rand_float(float min, float max) {
	float scale = (float) rand() / (float) RAND_MAX;
	return min + scale * (max - min);
}

// Column-major left-handed perspective projection (z in [0,1]), matching the
// original sample's XMMatrixPerspectiveFovLH -- GLSL's default matrix layout
// is column-major, so this array uploads directly as a mat4 uniform with no
// transpose.
static void build_perspective_lh(float fov_y_radians, float aspect, float z_near, float z_far, float out[16]) {
	float f = 1.0f / tanf(fov_y_radians * 0.5f);
	memset(out, 0, sizeof(float) * 16);
	out[0]  = f / aspect;
	out[5]  = f;
	out[10] = z_far / (z_far - z_near);
	out[11] = 1.0f;
	out[14] = -(z_far * z_near) / (z_far - z_near);
}

// NULL, printed, if the file can't be read.
static void *read_entire_file(const char *path, size_t *out_size) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open shader file: %s\n", path);
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	void *buffer = malloc((size_t) size);
	if (!buffer) {
		fprintf(stderr, "No memory for shader file: %s\n", path);
		fclose(f);
		return NULL;
	}
	fread(buffer, 1, (size_t) size, f);
	fclose(f);
	*out_size = (size_t) size;
	return buffer;
}

// NULL, printed, on failure.
static spudgpu_shader_module load_shader_module(
    spudgpu_device device,
    const char *path,
    SPUDGPU_SHADER_STAGE stage) {
	size_t code_size;
	void *code = read_entire_file(path, &code_size);
	if (!code)
		return NULL;

	spudgpu_shader_module_desc desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SHADER_MODULE_DESC, NULL},
	    .stage      = stage,
	    .spirv_code = code,
	    .spirv_size = code_size,
	};
	spudgpu_shader_module module = NULL;
	SPUDRESULT result            = spudgpu_create_shader_module(device, &desc, &module);
	free(code);
	if (SPUDFAIL(result)) {
		fprintf(stderr, "spudgpu_create_shader_module failed for %s: %d\n", path, result);
		return NULL;
	}
	return module;
}

int main(int argc, char **argv) {
	// --frames N: render N frames, then leave the loop as closing the window
	// does, through the same cleanup. Without it the sample runs until the
	// window is closed.
	// --device N: use the Nth adapter spudgpu_enumerate_adapters lists. 0 if
	// not given.
	uint64_t frame_limit  = 0;
	uint32_t device_index = 0;
	for (int i = 1; i + 1 < argc; ++i) {
		if (strcmp(argv[i], "--frames") == 0)
			frame_limit = strtoull(argv[i + 1], NULL, 10);
		else if (strcmp(argv[i], "--device") == 0)
			device_index = (uint32_t)strtoul(argv[i + 1], NULL, 10);
	}
	uint64_t frames_run = 0;

	// Every object main makes, NULL until it is made. A failure jumps to
	// cleanup, which destroys whatever exists: every destroy call takes NULL.
	int exit_code                                     = 1;
	SDL_Window *window                                = NULL;
	spudgpu_instance instance                         = NULL;
	spudgpu_device device                             = NULL;
	spudgpu_command_queue graphics_queue              = NULL;
	spudgpu_surface surface                           = NULL;
	spudgpu_swap_chain swap_chain                     = NULL;
	spudgpu_buffer vertex_buffer                      = NULL;
	spudgpu_buffer_view vertex_buffer_view            = NULL;
	spudgpu_buffer scene_buffer                       = NULL;
	spudgpu_buffer_view scene_buffer_view             = NULL;
	spudgpu_buffer projection_buffer                  = NULL;
	spudgpu_buffer input_commands_buffer              = NULL;
	spudgpu_buffer output_commands_buffer             = NULL;
	spudgpu_descriptor_set_layout compute_set_layout  = NULL;
	spudgpu_descriptor_pool compute_pool              = NULL;
	spudgpu_shader_module compute_module              = NULL;
	spudgpu_compute_pipeline compute_pipeline         = NULL;
	spudgpu_descriptor_set_layout graphics_set_layout = NULL;
	spudgpu_descriptor_pool graphics_pool             = NULL;
	spudgpu_shader_module vertex_module               = NULL;
	spudgpu_shader_module fragment_module             = NULL;
	spudgpu_shader_pipeline graphics_pipeline         = NULL;
	spudgpu_command_allocator command_allocator       = NULL;
	spudgpu_command_list cmd                          = NULL;

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	window = SDL_CreateWindow("SpudGPU ExecuteIndirect", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		goto cleanup;
	}

	const SPUD_ALLOCATOR gpu_allocator           = spud_allocator_system();
	const spudgpu_instance_desc gpu_instance_desc = {
	    .base                = {SPUD_STRUCT_TYPE_GPU_INSTANCE_DESC, NULL},
	    .allocator           = &gpu_allocator,
	    .application_name    = "SpudGPUExecuteIndirect",
	    .application_version = 1,
	    .engine_name         = "SpudGPUSamples",
	    .engine_version      = 1,
	};
	if (SPUDFAIL(spudgpu_create_instance(&gpu_instance_desc, &instance))) {
		fprintf(stderr, "spudgpu_create_instance failed\n");
		goto cleanup;
	}

	const spudgpu_adapter *adapters = NULL;
	uint32_t adapter_count          = 0;
	if (SPUDFAIL(spudgpu_enumerate_adapters(instance, &adapters, &adapter_count)) || adapter_count == 0) {
		fprintf(stderr, "spudgpu_enumerate_adapters failed or listed no adapters\n");
		goto cleanup;
	}
	if (device_index >= adapter_count) {
		fprintf(stderr, "--device %u: there are %u adapters\n", device_index, adapter_count);
		goto cleanup;
	}
	const spudgpu_adapter adapter = adapters[device_index];

	SPUDGPU_ADAPTER_PROPERTIES adapter_properties = {.base = {SPUD_STRUCT_TYPE_GPU_ADAPTER_PROPERTIES, NULL}};
	spudgpu_get_adapter_properties(adapter, &adapter_properties);
	printf("Using device: %s\n", adapter_properties.description);

	// A window to present to and one direct queue, and exactly the features
	// the sample uses: a device has only what it is created with.
	const spudgpu_device_desc device_desc = {
	    .base               = {SPUD_STRUCT_TYPE_GPU_DEVICE_DESC, NULL},
	    .required_features  = SPUDGPU_FEATURE_PRESENT | (adapter_properties.supported_features & SPUDGPU_FEATURE_MULTI_DRAW_INDIRECT),
	    .direct_queue_count = 1,
	    // What the sample's descriptor pools hold together, with room to
	    // spare: the caller's to choose, and refused past the adapter's
	    // limit.
	    .max_descriptors    = 1024,
	    .max_samplers       = 64,
	    .debug_name         = "SpudGPUExecuteIndirect",
	};
	if (SPUDFAIL(spudgpu_create_device(adapter, &device_desc, &device))) {
		fprintf(stderr, "spudgpu_create_device failed\n");
		goto cleanup;
	}
	spudgpu_get_command_queue(device, SPUDGPU_COMMAND_LIST_TYPE_DIRECT, 0, &graphics_queue);
	// One indirect call takes the adapter's most entries with
	// multi-draw-indirect, else one.
	const uint32_t max_draw_indirect_count =
	    (spudgpu_get_device_features(device) & SPUDGPU_FEATURE_MULTI_DRAW_INDIRECT) ? adapter_properties.max_draw_indirect_count : 1;

	surface = spudgpu_create_surface_from_sdl3(instance, window);
	if (!surface) {
		fprintf(stderr, "spudgpu_create_surface_from_sdl3 failed\n");
		goto cleanup;
	}

	spudgpu_swap_chain_desc swap_chain_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SWAP_CHAIN_DESC, NULL},
	    .surface = surface,
	    .queue   = graphics_queue,
	    .width   = WINDOW_WIDTH,
	    .height  = WINDOW_HEIGHT,
#if SPUDGPU_COMPILE_METAL_API
	    .buffer_count = 1,
#else
	    .buffer_count = 2,
#endif
	    .format          = SPUDGPU_FORMAT_B8G8R8A8_UNORM,
	    .present_mode    = SPUDGPU_PRESENT_MODE_FIFO,
	    .fullscreen_mode = SPUDGPU_FULLSCREEN_MODE_WINDOWED,
	};
	if (SPUDFAIL(spudgpu_create_swap_chain(device, &swap_chain_desc, &swap_chain))) {
		fprintf(stderr, "spudgpu_create_swap_chain failed\n");
		goto cleanup;
	}

	// Vertex buffer: one hardcoded triangle, instanced/drawn TRIANGLE_COUNT
	// times via indirect draw args -- each instance's animated offset/color
	// comes from scene_buffer, bound as a second, per-instance vertex buffer
	// (see graphics_pipeline_desc.vertex_bindings below).
	static const Vertex vertices[3] = {
	    {{0.0f, TRIANGLE_HALF_WIDTH, TRIANGLE_DEPTH}},
	    {{TRIANGLE_HALF_WIDTH, -TRIANGLE_HALF_WIDTH, TRIANGLE_DEPTH}},
	    {{-TRIANGLE_HALF_WIDTH, -TRIANGLE_HALF_WIDTH, TRIANGLE_DEPTH}},
	};

	spudgpu_buffer_desc vertex_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_VERTEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(vertices),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &vertex_buffer_desc, &vertex_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer (vertex) failed\n");
		goto cleanup;
	}
	void *vertex_mapped = NULL;
	spudgpu_map_buffer(vertex_buffer, 0, 0, &vertex_mapped);
	memcpy(vertex_mapped, vertices, sizeof(vertices));
	spudgpu_flush_buffer(vertex_buffer, 0, 0);
	spudgpu_unmap_buffer(vertex_buffer);

	spudgpu_buffer_view_desc vertex_buffer_view_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_VIEW_DESC, NULL},
	    .parent_buffer             = vertex_buffer,
	    .offset_from_parent_buffer = 0,
	    .stride                    = sizeof(Vertex),
	    .size                      = sizeof(vertices),
	};
	spudgpu_create_buffer_view(vertex_buffer, &vertex_buffer_view_desc, &vertex_buffer_view);

	// Per-triangle animated state -- UNIFORM (culling test, read by the
	// compute pass through a descriptor set) + VERTEX (offset/color, read by
	// the graphics pass as a per-instance vertex buffer -- see
	// graphics_pipeline_desc below). Kept mapped for the app's lifetime and
	// rewritten every frame, same pattern as HelloConstBuffers.
	SceneConstantBuffer scene_cb_data[TRIANGLE_COUNT];
	for (uint32_t i = 0; i < TRIANGLE_COUNT; i++) {
		scene_cb_data[i].velocity[0] = rand_float(0.01f, 0.02f);
		scene_cb_data[i].velocity[1] = scene_cb_data[i].velocity[2] = scene_cb_data[i].velocity[3] = 0.0f;
		scene_cb_data[i].offset[0]   = rand_float(-5.0f, -1.5f);
		scene_cb_data[i].offset[1]   = rand_float(-1.0f, 1.0f);
		scene_cb_data[i].offset[2]   = rand_float(0.0f, 2.0f);
		scene_cb_data[i].offset[3]   = 0.0f;
		scene_cb_data[i].color[0]    = rand_float(0.5f, 1.0f);
		scene_cb_data[i].color[1]    = rand_float(0.5f, 1.0f);
		scene_cb_data[i].color[2]    = rand_float(0.5f, 1.0f);
		scene_cb_data[i].color[3]    = 1.0f;
	}

	spudgpu_buffer_desc scene_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM | SPUDGPU_BUFFER_USAGE_VERTEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(scene_cb_data),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &scene_buffer_desc, &scene_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer (scene) failed\n");
		goto cleanup;
	}
	void *scene_mapped = NULL;
	spudgpu_map_buffer(scene_buffer, 0, 0, &scene_mapped);
	memcpy(scene_mapped, scene_cb_data, sizeof(scene_cb_data));
	spudgpu_flush_buffer(scene_buffer, 0, sizeof(scene_cb_data));

	spudgpu_buffer_view_desc scene_buffer_view_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_VIEW_DESC, NULL},
	    .parent_buffer             = scene_buffer,
	    .offset_from_parent_buffer = 0,
	    .stride                    = sizeof(SceneConstantBuffer),
	    .size                      = sizeof(scene_cb_data),
	};
	spudgpu_create_buffer_view(scene_buffer, &scene_buffer_view_desc, &scene_buffer_view);

	// Shared projection matrix -- identical for every triangle (fixed aspect
	// ratio/FOV), so it's one small uniform rather than duplicated
	// TRIANGLE_COUNT times the way the D3D12 original packs it.
	float projection[16];
	build_perspective_lh(
	    (float) M_PI / 4.0f, (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT, 0.01f, 20.0f, projection);

	spudgpu_buffer_desc projection_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(projection),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &projection_buffer_desc, &projection_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer (projection) failed\n");
		goto cleanup;
	}
	void *projection_mapped = NULL;
	spudgpu_map_buffer(projection_buffer, 0, 0, &projection_mapped);
	memcpy(projection_mapped, projection, sizeof(projection));
	spudgpu_flush_buffer(projection_buffer, 0, 0);
	spudgpu_unmap_buffer(projection_buffer);

	// Input draw-argument buffer: TRIANGLE_COUNT plain draw commands, built
	// once and never touched again -- first_instance bakes in the triangle's
	// index into scene_buffer (see shaders/scene.vert). UNIFORM (not STORAGE)
	// so it can be UPLOAD memory and written directly (UPLOAD refuses
	// STORAGE on every backend: the GPU only reads it), plus INDIRECT so it
	// can be
	// read directly by spudgpu_cmd_draw_indirect when culling is off.
	IndirectCommand input_commands[TRIANGLE_COUNT];
	for (uint32_t i = 0; i < TRIANGLE_COUNT; i++) {
		input_commands[i].vertex_count   = 3;
		input_commands[i].instance_count = 1;
		input_commands[i].first_vertex   = 0;
		input_commands[i].first_instance = i;
	}

	spudgpu_buffer_desc input_commands_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM | SPUDGPU_BUFFER_USAGE_INDIRECT,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(input_commands),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &input_commands_desc, &input_commands_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer (input commands) failed\n");
		goto cleanup;
	}
	void *input_commands_mapped = NULL;
	spudgpu_map_buffer(input_commands_buffer, 0, 0, &input_commands_mapped);
	memcpy(input_commands_mapped, input_commands, sizeof(input_commands));
	spudgpu_flush_buffer(input_commands_buffer, 0, 0);
	spudgpu_unmap_buffer(input_commands_buffer);

	// Output draw-argument buffer: written by the compute pass every frame
	// (real args for a visible triangle, zeroed vertex_count for a culled
	// one), then read directly as the indirect-draw source. STORAGE (the
	// compute shader writes it) + INDIRECT, in DEVICE memory -- it's never
	// touched from the CPU, so nothing needs it mapped.
	spudgpu_buffer_desc output_commands_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_STORAGE | SPUDGPU_BUFFER_USAGE_INDIRECT,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_DEVICE,
	    .size         = sizeof(input_commands),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &output_commands_desc, &output_commands_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer (output commands) failed\n");
		goto cleanup;
	}

	// Compute descriptor set: scene (0), input commands (1), output commands
	// (2, read/write), projection (3).
	spudgpu_descriptor_set_layout_desc compute_set_layout_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_LAYOUT_DESC, NULL},
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_COMPUTE},
	        {.binding = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_COMPUTE},
	        {.binding = 2, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_COMPUTE},
	        {.binding = 3, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_COMPUTE},
	    },
	    .binding_count = 4,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_set_layout(device, &compute_set_layout_desc, &compute_set_layout))) {
		fprintf(stderr, "spudgpu_create_descriptor_set_layout (compute) failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_pool_desc compute_pool_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_POOL_DESC, NULL},
	    .max_sets   = 1,
	    .pool_sizes = {
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 3},
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1},
	    },
	    .pool_size_count = 2,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_pool(device, &compute_pool_desc, &compute_pool))) {
		fprintf(stderr, "spudgpu_create_descriptor_pool (compute) failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_set_desc compute_set_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_DESC, NULL},
	    .pool        = compute_pool,
	    .set_layouts = {compute_set_layout},
	    .set_count   = 1,
	};
	spudgpu_descriptor_set compute_set = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_sets(device, &compute_set_desc, &compute_set))) {
		fprintf(stderr, "spudgpu_create_descriptor_sets (compute) failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_buffer_info compute_scene_info      = {.buffer = scene_buffer, .offset = 0, .range = sizeof(scene_cb_data)};
	spudgpu_descriptor_buffer_info compute_input_info      = {.buffer = input_commands_buffer, .offset = 0, .range = sizeof(input_commands)};
	spudgpu_descriptor_buffer_info compute_output_info     = {.buffer = output_commands_buffer, .offset = 0, .range = sizeof(input_commands)};
	spudgpu_descriptor_buffer_info compute_projection_info = {.buffer = projection_buffer, .offset = 0, .range = sizeof(projection)};
	spudgpu_write_descriptor_set compute_writes[4] = {
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = compute_set, .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer_info = &compute_scene_info},
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = compute_set, .dst_binding = 1, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer_info = &compute_input_info},
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = compute_set, .dst_binding = 2, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = &compute_output_info},
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = compute_set, .dst_binding = 3, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer_info = &compute_projection_info},
	};
	spudgpu_update_descriptor_sets(device, compute_writes, 4);

	compute_module = load_shader_module(device, "shaders/cull.comp.spv", SPUDGPU_SHADER_STAGE_COMPUTE);
	if (!compute_module)
		goto cleanup;

	spudgpu_compute_pipeline_desc compute_pipeline_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_COMPUTE_PIPELINE_DESC, NULL},
	    .compute_module              = compute_module,
	    .descriptor_set_layouts      = {compute_set_layout},
	    .descriptor_set_layout_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_compute_pipeline(device, &compute_pipeline_desc, &compute_pipeline))) {
		fprintf(stderr, "spudgpu_create_compute_pipeline failed\n");
		goto cleanup;
	}

	// Graphics descriptor set: projection (0). Per-triangle offset/color no
	// longer goes through a descriptor -- it's a per-instance vertex buffer
	// (see graphics_pipeline_desc below).
	spudgpu_descriptor_set_layout_desc graphics_set_layout_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_LAYOUT_DESC, NULL},
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_VERTEX},
	    },
	    .binding_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_set_layout(device, &graphics_set_layout_desc, &graphics_set_layout))) {
		fprintf(stderr, "spudgpu_create_descriptor_set_layout (graphics) failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_pool_desc graphics_pool_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_POOL_DESC, NULL},
	    .max_sets        = 1,
	    .pool_sizes      = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1}},
	    .pool_size_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_pool(device, &graphics_pool_desc, &graphics_pool))) {
		fprintf(stderr, "spudgpu_create_descriptor_pool (graphics) failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_set_desc graphics_set_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_DESC, NULL},
	    .pool        = graphics_pool,
	    .set_layouts = {graphics_set_layout},
	    .set_count   = 1,
	};
	spudgpu_descriptor_set graphics_set = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_sets(device, &graphics_set_desc, &graphics_set))) {
		fprintf(stderr, "spudgpu_create_descriptor_sets (graphics) failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_buffer_info graphics_projection_info = {.buffer = projection_buffer, .offset = 0, .range = sizeof(projection)};
	spudgpu_write_descriptor_set graphics_writes[1] = {
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = graphics_set, .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer_info = &graphics_projection_info},
	};
	spudgpu_update_descriptor_sets(device, graphics_writes, 1);

	vertex_module   = load_shader_module(device, "shaders/scene.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	fragment_module = load_shader_module(device, "shaders/scene.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	if (!vertex_module || !fragment_module)
		goto cleanup;

	spudgpu_shader_pipeline_desc graphics_pipeline_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SHADER_PIPELINE_DESC, NULL},
	    .vertex_module      = vertex_module,
	    .fragment_module    = fragment_module,
	    .vertex_attributes  = {
	        {.location = 0, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, position)},
	        {.location = 1, .binding = 1, .format = SPUDGPU_FORMAT_R32G32B32A32_FLOAT, .offset = offsetof(SceneConstantBuffer, offset)},
	        {.location = 2, .binding = 1, .format = SPUDGPU_FORMAT_R32G32B32A32_FLOAT, .offset = offsetof(SceneConstantBuffer, color)},
	    },
	    .vertex_attribute_count = 3,
	    .vertex_bindings        = {
	        {.binding = 0, .stride = sizeof(Vertex), .per_instance = false},
	        {.binding = 1, .stride = sizeof(SceneConstantBuffer), .per_instance = true},
	    },
	    .vertex_binding_count        = 2,
	    .primitive_topology          = SPUDGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	    .cull_mode                   = SPUDGPU_CULL_MODE_NONE,
	    .front_face_ccw              = true,
	    .color_attachment_formats    = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count      = 1,
	    .depth_format                = SPUDGPU_FORMAT_UNKNOWN,
	    .descriptor_set_layouts      = {graphics_set_layout},
	    .descriptor_set_layout_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &graphics_pipeline_desc, &graphics_pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		goto cleanup;
	}

	spudgpu_command_allocator_desc allocator_desc = {.base = {SPUD_STRUCT_TYPE_GPU_COMMAND_ALLOCATOR_DESC, NULL}, .type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_create_command_list(command_allocator, &cmd);

	const uint32_t dispatch_group_count = (TRIANGLE_COUNT + COMPUTE_THREAD_BLOCK_SIZE - 1) / COMPUTE_THREAD_BLOCK_SIZE;
	const float offset_bounds           = 2.5f;

	SPUDGPU_SCISSOR_RECT full_scissor = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};
	SPUDGPU_SCISSOR_RECT culling_scissor = {
	    .x      = (WINDOW_WIDTH / 2.0f) - (WINDOW_WIDTH / 2.0f) * CULL_OFFSET,
	    .y      = 0,
	    .width  = WINDOW_WIDTH * CULL_OFFSET,
	    .height = WINDOW_HEIGHT,
	};

	bool running        = true;
	bool enable_culling = true;
	while (running) {
		if (frame_limit != 0 && frames_run == frame_limit)
			break;
		++frames_run;
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
			if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_SPACE && !event.key.repeat)
				enable_culling = !enable_culling;
		}

		for (uint32_t i = 0; i < TRIANGLE_COUNT; i++) {
			scene_cb_data[i].offset[0] += scene_cb_data[i].velocity[0];
			if (scene_cb_data[i].offset[0] > offset_bounds) {
				scene_cb_data[i].velocity[0] = rand_float(0.01f, 0.02f);
				scene_cb_data[i].offset[0]   = -offset_bounds;
			}
		}
		memcpy(scene_mapped, scene_cb_data, sizeof(scene_cb_data));
		spudgpu_flush_buffer(scene_buffer, 0, sizeof(scene_cb_data));

		uint32_t image_index               = spudgpu_swap_chain_acquire_next_image(swap_chain);
		spudgpu_image_view backbuffer_view = spudgpu_get_swap_chain_image_view(swap_chain, image_index);

		spudgpu_reset_command_allocator(command_allocator);
		spudgpu_begin_command_list(cmd);

		if (enable_culling) {
			spudgpu_cmd_bind_compute_pipeline(cmd, compute_pipeline);
			spudgpu_cmd_bind_descriptor_sets_compute(cmd, compute_pipeline, 0, &compute_set, 1);
			spudgpu_cmd_dispatch(cmd, dispatch_group_count, 1, 1);

			spudgpu_buffer_barrier post_dispatch_barrier = {
			    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_BARRIER, NULL},
			    .buffer       = output_commands_buffer,
			    .state_before = SPUDGPU_RESOURCE_STATE_UNORDERED_ACCESS,
			    .state_after  = SPUDGPU_RESOURCE_STATE_INDIRECT_ARGUMENT,
			};
			spudgpu_cmd_pipeline_barrier(cmd, &post_dispatch_barrier, 1, NULL, 0);
		}

		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		spudgpu_rendering_begin_desc rendering_desc = {
		    .base = {SPUD_STRUCT_TYPE_GPU_RENDERING_BEGIN_DESC, NULL},
		    .color_attachments = {
		        {
		            .image_view  = backbuffer_view,
		            .load_op     = SPUDGPU_LOAD_OP_CLEAR,
		            .store_op    = SPUDGPU_STORE_OP_STORE,
		            .clear_color = {0.0f, 0.02f, 0.04f, 1.0f},
		        },
		    },
		    .color_attachment_count = 1,
		    .width                  = WINDOW_WIDTH,
		    .height                 = WINDOW_HEIGHT,
		};
		spudgpu_cmd_begin_rendering(cmd, &rendering_desc);

		spudgpu_cmd_bind_pipeline(cmd, graphics_pipeline);
		spudgpu_cmd_bind_descriptor_sets(cmd, graphics_pipeline, 0, &graphics_set, 1);

		SPUDGPU_VIEWPORT viewport = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT, .minDepth = 0.0f, .maxDepth = 1.0f};
		spudgpu_cmd_set_viewports(cmd, 0, 1, &viewport);
		spudgpu_cmd_set_scissor_rects(cmd, 0, 1, enable_culling ? &culling_scissor : &full_scissor);

		spudgpu_buffer_view graphics_vertex_buffer_views[2] = {vertex_buffer_view, scene_buffer_view};
		spudgpu_cmd_set_vertex_buffers(cmd, 0, 2, graphics_vertex_buffer_views);
		// One call where the device takes TRIANGLE_COUNT draws in one (a
		// device with SPUDGPU_FEATURE_MULTI_DRAW_INDIRECT), else as many calls
		// as its max_draw_indirect_count needs, at advancing offsets.
		for (uint32_t first = 0; first < TRIANGLE_COUNT;) {
			uint32_t left  = TRIANGLE_COUNT - first;
			uint32_t count = left < max_draw_indirect_count ? left : max_draw_indirect_count;
			spudgpu_cmd_draw_indirect(
			    cmd,
			    enable_culling ? output_commands_buffer : input_commands_buffer,
			    (uint64_t)first * sizeof(IndirectCommand), count, sizeof(IndirectCommand));
			first += count;
		}

		spudgpu_cmd_end_rendering(cmd);
		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, SPUDGPU_IMAGE_LAYOUT_PRESENT_SRC);

		if (enable_culling) {
			// Ready output_commands_buffer for next frame's compute write.
			spudgpu_buffer_barrier pre_dispatch_barrier = {
			    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_BARRIER, NULL},
			    .buffer       = output_commands_buffer,
			    .state_before = SPUDGPU_RESOURCE_STATE_INDIRECT_ARGUMENT,
			    .state_after  = SPUDGPU_RESOURCE_STATE_UNORDERED_ACCESS,
			};
			spudgpu_cmd_pipeline_barrier(cmd, &pre_dispatch_barrier, 1, NULL, 0);
		}

		spudgpu_end_command_list(cmd);

		spudgpu_submit_command_lists_synced(graphics_queue, &cmd, 1, swap_chain);
		spudgpu_swap_chain_present(swap_chain);
	}

	exit_code = 0;

cleanup:
	if (graphics_queue)
		spudgpu_queue_wait_idle(graphics_queue);

	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_shader_pipeline(graphics_pipeline);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(vertex_module);
	spudgpu_destroy_descriptor_pool(graphics_pool);
	spudgpu_destroy_descriptor_set_layout(graphics_set_layout);
	spudgpu_destroy_compute_pipeline(compute_pipeline);
	spudgpu_destroy_shader_module(compute_module);
	spudgpu_destroy_descriptor_pool(compute_pool);
	spudgpu_destroy_descriptor_set_layout(compute_set_layout);
	if (scene_buffer)
		spudgpu_unmap_buffer(scene_buffer);
	spudgpu_destroy_buffer_view(scene_buffer_view);
	spudgpu_destroy_buffer(scene_buffer);
	spudgpu_destroy_buffer(projection_buffer);
	spudgpu_destroy_buffer(input_commands_buffer);
	spudgpu_destroy_buffer(output_commands_buffer);
	spudgpu_destroy_buffer_view(vertex_buffer_view);
	spudgpu_destroy_buffer(vertex_buffer);
	spudgpu_destroy_swap_chain(swap_chain);
	spudgpu_destroy_surface(surface);
	spudgpu_destroy_device(device);
	spudgpu_destroy_instance(instance);

	if (window)
		SDL_DestroyWindow(window);
	SDL_Quit();
	return exit_code;
}
