//
// SpudGPU port of D3D12Bundles from d3d12samples. A CITY_ROW_COUNT x
// CITY_COLUMN_COUNT grid of "occcity" building meshes (the same asset
// SpudGPUDynamicIndexing uses), each with its own per-object CBV. The point
// of the original sample -- and this port -- isn't the city itself, it's
// comparing two ways of getting the same 30 draw calls to the GPU every
// frame:
//
//   - Bundle mode (the default): the whole per-object bind+draw sequence is
//     recorded ONCE, before the render loop starts, into a
//     SPUDGPU_COMMAND_LIST_TYPE_BUNDLE command list, then replayed every
//     frame with a single spudgpu_cmd_execute_bundle call. Only the CBV
//     buffer's bytes change frame to frame (fresh view-projection matrices);
//     the recorded commands referencing those CBVs never do.
//   - Direct mode: the exact same per-object bind+draw sequence is
//     re-recorded from scratch into the direct command list every single
//     frame.
//
// Press C to toggle between them -- matches the original sample's own
// keybinding. This closed a real gap in spudlib: SPUDGPU_COMMAND_LIST_TYPE_
// BUNDLE already existed as an enum value, but nothing could actually
// execute one -- see SPUDGPU_EXT_BUNDLES in spudgpu.h and ../../README.md
// for what that took on each backend, and why Metal can't have this sample
// at all (a bundle needs CPU-side reusable secondary command recording,
// which Metal's single-use MTLCommandBuffer/MTLRenderCommandEncoder
// structurally cannot do).
//
// Two deliberate simplifications versus the original sample, in the same
// spirit as this suite's other ports (see ../../README.md):
//   - No diffuse texture / bindless material grid (that's
//     SpudGPUDynamicIndexing's whole point, already covered there). Each
//     building instead gets a fixed HSL-gradient color baked into its CBV
//     alongside its MVP matrix, so this sample only exercises the bundle
//     mechanism itself, not texture sampling on top of it.
//   - Single-buffered CBV storage (one persistently-mapped buffer,
//     overwritten in place every frame), matching every other sample in
//     this suite (none of them triple-buffer per-frame resources) rather
//     than the original's 3 rotating FrameResources. The bundle's recorded
//     descriptor-set bindings point at fixed buffer offsets regardless --
//     only the bytes at those offsets change per frame -- so this
//     simplification doesn't touch the actual mechanism being demonstrated.
//
// occcity.bin is copied verbatim from
// d3d12samples/Samples/Desktop/D3D12DynamicIndexing/src/ (the same asset
// D3D12Bundles itself uses, under its own SampleAssets namespace) -- the
// fixed byte offsets/sizes below come from that sample's occcity.h.
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

#if !SPUDGPU_EXT_BUNDLES
#error "SpudGPUBundles needs SPUDGPU_EXT_BUNDLES (Vulkan/D3D12 only -- see spudlib/CLAUDE.md and ../../README.md)."
#endif

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

// The one viewport and scissor every draw uses: named to the bundle when it
// is recorded, and set on the direct list each frame before it executes the
// bundle, as spudgpu.h's bundle rules require the two to match.
static const SPUDGPU_VIEWPORT window_viewport = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT, .minDepth = 0.0f, .maxDepth = 1.0f};
static const SPUDGPU_SCISSOR_RECT window_scissor = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};

// Matches the original D3D12Bundles' CityRowCount/CityColumnCount exactly.
#define CITY_ROW_COUNT 10
#define CITY_COLUMN_COUNT 3
#define CITY_OBJECT_COUNT (CITY_ROW_COUNT * CITY_COLUMN_COUNT) // 30
#define CITY_SPACING_INTERVAL 16.0f

// occcity.bin layout, from d3d12samples' occcity.h (SampleAssets namespace).
// Only the mesh portion is used here -- see the file comment above for why
// this port skips the diffuse texture entirely.
#define VERTEX_DATA_OFFSET 524288u
#define VERTEX_DATA_SIZE 820248u
#define INDEX_DATA_OFFSET 1344536u
#define INDEX_DATA_SIZE 74568u
#define VERTEX_STRIDE 44u

#define SPUDGPU_ASPECT_DEPTH_BIT 0x00000002u

typedef struct Vertex {
	float position[3];
	float normal[3]; // Feeds basic directional lighting in scene.frag.
	float uv[2];      // Unused (no texturing).
	float tangent[3]; // Unused.
} Vertex;

// Matches shaders/scene.vert's CBV block exactly (std140: mat4 + vec4 need
// no manual padding). One per building; mvp is rewritten every frame, color
// is written once at init and never touched again.
typedef struct ObjectCB {
	float mvp[16];
	float color[4];
} ObjectCB;

// ---------------------------------------------------------------------------
// Math (column-major, right-handed -- same convention as
// SpudGPUDynamicIndexing, which this sample's camera/grid code is ported
// from almost verbatim).
// ---------------------------------------------------------------------------

static void vec3_cross(const float a[3], const float b[3], float out[3]) {
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

static float vec3_dot(const float a[3], const float b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void vec3_normalize(float v[3]) {
	float len = sqrtf(vec3_dot(v, v));
	if (len > 1e-6f) {
		v[0] /= len;
		v[1] /= len;
		v[2] /= len;
	}
}

static void mat4_multiply(const float a[16], const float b[16], float out[16]) {
	float result[16];
	for (int col = 0; col < 4; col++) {
		for (int row = 0; row < 4; row++) {
			float sum = 0.0f;
			for (int k = 0; k < 4; k++)
				sum += a[k * 4 + row] * b[col * 4 + k];
			result[col * 4 + row] = sum;
		}
	}
	memcpy(out, result, sizeof(result));
}

static void mat4_translation(float x, float y, float z, float out[16]) {
	memset(out, 0, sizeof(float) * 16);
	out[0] = out[5] = out[10] = out[15] = 1.0f;
	out[12] = x;
	out[13] = y;
	out[14] = z;
}

static void mat4_look_to_rh(const float eye[3], const float dir[3], const float up[3], float out[16]) {
	float zaxis[3] = {-dir[0], -dir[1], -dir[2]};
	vec3_normalize(zaxis);
	float xaxis[3];
	vec3_cross(up, zaxis, xaxis);
	vec3_normalize(xaxis);
	float yaxis[3];
	vec3_cross(zaxis, xaxis, yaxis);

	out[0] = xaxis[0];
	out[1] = yaxis[0];
	out[2] = zaxis[0];
	out[3] = 0.0f;
	out[4] = xaxis[1];
	out[5] = yaxis[1];
	out[6] = zaxis[1];
	out[7] = 0.0f;
	out[8] = xaxis[2];
	out[9] = yaxis[2];
	out[10] = zaxis[2];
	out[11] = 0.0f;
	out[12] = -vec3_dot(xaxis, eye);
	out[13] = -vec3_dot(yaxis, eye);
	out[14] = -vec3_dot(zaxis, eye);
	out[15] = 1.0f;
}

static void mat4_perspective_rh(float fov_y_radians, float aspect, float z_near, float z_far, float out[16]) {
	float f = 1.0f / tanf(fov_y_radians * 0.5f);
	memset(out, 0, sizeof(float) * 16);
	out[0] = f / aspect;
	out[5] = f;
	out[10] = z_far / (z_near - z_far);
	out[11] = -1.0f;
	out[14] = (z_near * z_far) / (z_near - z_far);
}

// ---------------------------------------------------------------------------
// HSL -> RGB, for each building's fixed per-object color (a portable stand-in
// for the original's per-building diffuse texture -- see the file comment
// above).
// ---------------------------------------------------------------------------

static float hsl_hue_to_rgb(float p, float q, float t) {
	if (t < 0.0f) t += 1.0f;
	if (t > 1.0f) t -= 1.0f;
	if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
	if (t < 1.0f / 2.0f) return q;
	if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
	return p;
}

static void hsl_to_rgb_f(float h, float s, float l, float out[3]) {
	if (s == 0.0f) {
		out[0] = out[1] = out[2] = l;
		return;
	}
	float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
	float p = 2.0f * l - q;
	out[0]  = hsl_hue_to_rgb(p, q, h + 1.0f / 3.0f);
	out[1]  = hsl_hue_to_rgb(p, q, h);
	out[2]  = hsl_hue_to_rgb(p, q, h - 1.0f / 3.0f);
}

// ---------------------------------------------------------------------------
// Camera (WASD move, arrow-key look) -- ported verbatim from
// SpudGPUDynamicIndexing.
// ---------------------------------------------------------------------------

typedef struct Camera {
	float position[3];
	float yaw, pitch;
	float look_dir[3];
	float move_speed;
	float turn_speed;
	bool key_w, key_a, key_s, key_d;
	bool key_left, key_right, key_up, key_down;
} Camera;

static void camera_init(Camera *cam, float x, float y, float z, float move_speed) {
	memset(cam, 0, sizeof(*cam));
	cam->position[0] = x;
	cam->position[1] = y;
	cam->position[2] = z;
	cam->yaw          = (float) M_PI;
	cam->pitch        = 0.0f;
	cam->look_dir[2]  = -1.0f;
	cam->move_speed   = move_speed;
	cam->turn_speed   = (float) M_PI / 2.0f;
}

static void camera_update(Camera *cam, float dt) {
	float move_x = 0.0f, move_z = 0.0f;
	if (cam->key_a) move_x -= 1.0f;
	if (cam->key_d) move_x += 1.0f;
	if (cam->key_w) move_z -= 1.0f;
	if (cam->key_s) move_z += 1.0f;
	if (fabsf(move_x) > 0.1f && fabsf(move_z) > 0.1f) {
		float inv_len = 1.0f / sqrtf(move_x * move_x + move_z * move_z);
		move_x *= inv_len;
		move_z *= inv_len;
	}

	float move_interval   = cam->move_speed * dt;
	float rotate_interval = cam->turn_speed * dt;

	if (cam->key_left) cam->yaw += rotate_interval;
	if (cam->key_right) cam->yaw -= rotate_interval;
	if (cam->key_up) cam->pitch += rotate_interval;
	if (cam->key_down) cam->pitch -= rotate_interval;

	const float pitch_limit = (float) M_PI / 4.0f;
	if (cam->pitch > pitch_limit) cam->pitch = pitch_limit;
	if (cam->pitch < -pitch_limit) cam->pitch = -pitch_limit;

	float x = move_x * -cosf(cam->yaw) - move_z * sinf(cam->yaw);
	float z = move_x * sinf(cam->yaw) - move_z * cosf(cam->yaw);
	cam->position[0] += x * move_interval;
	cam->position[2] += z * move_interval;

	float r = cosf(cam->pitch);
	cam->look_dir[0] = r * sinf(cam->yaw);
	cam->look_dir[1] = sinf(cam->pitch);
	cam->look_dir[2] = r * cosf(cam->yaw);
}

// NULL, printed, if the file can't be read.
static void *read_entire_file(const char *path, size_t *out_size) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open file: %s\n", path);
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	void *buffer = malloc((size_t) size);
	if (!buffer) {
		fprintf(stderr, "No memory for file: %s\n", path);
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

// Records the fixed sequence every building's draw needs: bind the pipeline,
// vertex/index buffers, then per building bind its CBV descriptor set and
// draw. Called exactly once, into the bundle, in bundle mode; called every
// frame, into the direct command list, in direct mode -- see the file
// comment above.
//
// Viewport/scissor are NOT set here -- D3D12 bundles inherit both from the
// calling command list and error on RSSetViewports/RSSetScissorRects
// (EXECUTION ERROR #546: INVALID_BUNDLE_API); the caller sets them on cmd
// once per frame instead, before either executing the bundle or calling
// this function directly. A Vulkan secondary buffer inherits neither, so
// the bundle is also given them in its spudgpu_bundle_inheritance_desc.
static void record_city_draws(
    spudgpu_command_list target,
    spudgpu_shader_pipeline pipeline,
    spudgpu_buffer_view vertex_buffer_view,
    spudgpu_buffer_view index_buffer_view,
    uint32_t index_count,
    spudgpu_descriptor_set object_sets[CITY_OBJECT_COUNT]) {
	spudgpu_cmd_bind_pipeline(target, pipeline);

	spudgpu_cmd_set_vertex_buffers(target, 0, 1, &vertex_buffer_view);
	spudgpu_cmd_set_index_buffer(target, index_buffer_view);

	for (uint32_t i = 0; i < CITY_OBJECT_COUNT; i++) {
		spudgpu_cmd_bind_descriptor_sets(target, pipeline, 0, &object_sets[i], 1);
		spudgpu_cmd_draw_indexed(target, index_count, 0, 0);
	}
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
	int exit_code                               = 1;
	SDL_Window *window                          = NULL;
	spudgpu_instance instance                   = NULL;
	spudgpu_device device                       = NULL;
	spudgpu_command_queue graphics_queue        = NULL;
	spudgpu_surface surface                     = NULL;
	spudgpu_swap_chain swap_chain               = NULL;
	spudgpu_command_allocator command_allocator = NULL;
	spudgpu_command_list cmd                    = NULL;
	spudgpu_buffer vertex_buffer                = NULL;
	spudgpu_buffer_view vertex_buffer_view      = NULL;
	spudgpu_buffer index_buffer                 = NULL;
	spudgpu_buffer_view index_buffer_view       = NULL;
	spudgpu_descriptor_set_layout cbv_layout    = NULL;
	spudgpu_descriptor_pool cbv_pool            = NULL;
	spudgpu_buffer cbv_buffer                   = NULL;
	spudgpu_shader_module vertex_module         = NULL;
	spudgpu_shader_module fragment_module       = NULL;
	spudgpu_shader_pipeline pipeline            = NULL;
	spudgpu_image depth_image                   = NULL;
	spudgpu_image_view depth_view               = NULL;
	spudgpu_command_allocator bundle_allocator  = NULL;
	spudgpu_command_list bundle                 = NULL;

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	window = SDL_CreateWindow("SpudGPU Bundles [Bundles ON]", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		goto cleanup;
	}

	const SPUD_ALLOCATOR gpu_allocator           = spud_allocator_system();
	const spudgpu_instance_desc gpu_instance_desc = {
	    .base                = {SPUD_STRUCT_TYPE_GPU_INSTANCE_DESC, NULL},
	    .allocator           = &gpu_allocator,
	    .application_name    = "SpudGPUBundles",
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
	    .required_features  = SPUDGPU_FEATURE_PRESENT,
	    .direct_queue_count = 1,
	    // What the sample's descriptor pools hold together, with room to
	    // spare: the caller's to choose, and refused past the adapter's
	    // limit.
	    .max_descriptors    = 1024,
	    .max_samplers       = 64,
	    .debug_name         = "SpudGPUBundles",
	};
	if (SPUDFAIL(spudgpu_create_device(adapter, &device_desc, &device))) {
		fprintf(stderr, "spudgpu_create_device failed\n");
		goto cleanup;
	}
	spudgpu_get_command_queue(device, SPUDGPU_COMMAND_LIST_TYPE_DIRECT, 0, &graphics_queue);

	surface = spudgpu_create_surface_from_sdl3(instance, window);
	if (!surface) {
		fprintf(stderr, "spudgpu_create_surface_from_sdl3 failed\n");
		goto cleanup;
	}

	spudgpu_swap_chain_desc swap_chain_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SWAP_CHAIN_DESC, NULL},
	    .surface         = surface,
	    .queue           = graphics_queue,
	    .width           = WINDOW_WIDTH,
	    .height          = WINDOW_HEIGHT,
	    .buffer_count    = 2,
	    .format          = SPUDGPU_FORMAT_B8G8R8A8_UNORM,
	    .present_mode    = SPUDGPU_PRESENT_MODE_FIFO,
	    .fullscreen_mode = SPUDGPU_FULLSCREEN_MODE_WINDOWED,
	};
	if (SPUDFAIL(spudgpu_create_swap_chain(device, &swap_chain_desc, &swap_chain))) {
		fprintf(stderr, "spudgpu_create_swap_chain failed\n");
		goto cleanup;
	}

	spudgpu_command_allocator_desc direct_allocator_desc = {.base = {SPUD_STRUCT_TYPE_GPU_COMMAND_ALLOCATOR_DESC, NULL}, .type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_create_command_allocator(device, &direct_allocator_desc, &command_allocator);

	spudgpu_create_command_list(command_allocator, &cmd);

	// ------------------------------------------------------------------
	// Mesh (occcity.bin) -- vertex/index data only, no diffuse texture.
	// ------------------------------------------------------------------

	size_t mesh_data_size;
	uint8_t *mesh_data = (uint8_t *) read_entire_file("occcity.bin", &mesh_data_size);
	if (!mesh_data)
		goto cleanup;

	spudgpu_buffer_desc vertex_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_VERTEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = VERTEX_DATA_SIZE,
	};
	spudgpu_create_buffer(device, &vertex_buffer_desc, &vertex_buffer);
	void *vertex_mapped = NULL;
	spudgpu_map_buffer(vertex_buffer, 0, 0, &vertex_mapped);
	memcpy(vertex_mapped, mesh_data + VERTEX_DATA_OFFSET, VERTEX_DATA_SIZE);
	spudgpu_flush_buffer(vertex_buffer, 0, 0);
	spudgpu_unmap_buffer(vertex_buffer);

	spudgpu_buffer_view_desc vertex_view_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_VIEW_DESC, NULL},
	    .parent_buffer = vertex_buffer, .offset_from_parent_buffer = 0, .stride = VERTEX_STRIDE, .size = VERTEX_DATA_SIZE};
	spudgpu_create_buffer_view(vertex_buffer, &vertex_view_desc, &vertex_buffer_view);

	spudgpu_buffer_desc index_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_INDEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = INDEX_DATA_SIZE,
	};
	spudgpu_create_buffer(device, &index_buffer_desc, &index_buffer);
	void *index_mapped = NULL;
	spudgpu_map_buffer(index_buffer, 0, 0, &index_mapped);
	memcpy(index_mapped, mesh_data + INDEX_DATA_OFFSET, INDEX_DATA_SIZE);
	spudgpu_flush_buffer(index_buffer, 0, 0);
	spudgpu_unmap_buffer(index_buffer);
	free(mesh_data);

	spudgpu_buffer_view_desc index_view_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_VIEW_DESC, NULL},
	    .parent_buffer = index_buffer, .offset_from_parent_buffer = 0, .stride = 4, .size = INDEX_DATA_SIZE};
	spudgpu_create_buffer_view(index_buffer, &index_view_desc, &index_buffer_view);

	uint32_t index_count = INDEX_DATA_SIZE / 4;

	// ------------------------------------------------------------------
	// Per-object CBVs: one descriptor set + buffer slice per building.
	// ------------------------------------------------------------------

	spudgpu_descriptor_set_layout_desc cbv_layout_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_LAYOUT_DESC, NULL},
	    .bindings      = {{.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_VERTEX}},
	    .binding_count = 1,
	};
	spudgpu_create_descriptor_set_layout(device, &cbv_layout_desc, &cbv_layout);

	spudgpu_descriptor_pool_desc cbv_pool_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_POOL_DESC, NULL},
	    .max_sets = CITY_OBJECT_COUNT, .pool_sizes = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = CITY_OBJECT_COUNT}}, .pool_size_count = 1};
	spudgpu_create_descriptor_pool(device, &cbv_pool_desc, &cbv_pool);

	const uint32_t cbv_stride = 256; // D3D12 CBV alignment; harmless elsewhere.
	spudgpu_buffer_desc cbv_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = (uint64_t) cbv_stride * CITY_OBJECT_COUNT,
	};
	spudgpu_create_buffer(device, &cbv_buffer_desc, &cbv_buffer);
	void *cbv_mapped = NULL;
	spudgpu_map_buffer(cbv_buffer, 0, 0, &cbv_mapped);

	spudgpu_descriptor_set object_sets[CITY_OBJECT_COUNT];
	float model_matrices[CITY_OBJECT_COUNT][16];
	for (uint32_t row = 0; row < CITY_ROW_COUNT; row++) {
		for (uint32_t col = 0; col < CITY_COLUMN_COUNT; col++) {
			uint32_t i = row * CITY_COLUMN_COUNT + col;

			mat4_translation(col * CITY_SPACING_INTERVAL, 0.0f, -(float) row * CITY_SPACING_INTERVAL, model_matrices[i]);

			spudgpu_descriptor_set_desc set_desc = {.base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_DESC, NULL}, .pool = cbv_pool, .set_layouts = {cbv_layout}, .set_count = 1};
			spudgpu_create_descriptor_sets(device, &set_desc, &object_sets[i]);

			spudgpu_descriptor_buffer_info buf_info = {.buffer = cbv_buffer, .offset = (uint64_t) i * cbv_stride, .range = sizeof(ObjectCB)};
			spudgpu_write_descriptor_set write = {
			    .base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL},
			    .dst_set = object_sets[i], .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer_info = &buf_info};
			spudgpu_update_descriptor_sets(device, &write, 1);

			// Color is written once here and never touched again -- only
			// mvp (below, every frame) changes.
			float color[3];
			hsl_to_rgb_f((float) i / (float) CITY_OBJECT_COUNT, 0.6f, 0.55f, color);
			ObjectCB *cb = (ObjectCB *) ((uint8_t *) cbv_mapped + i * cbv_stride);
			cb->color[0] = color[0];
			cb->color[1] = color[1];
			cb->color[2] = color[2];
			cb->color[3] = 1.0f;
		}
	}

	// ------------------------------------------------------------------
	// Pipeline + depth buffer.
	// ------------------------------------------------------------------

	vertex_module   = load_shader_module(device, "shaders/scene.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	fragment_module = load_shader_module(device, "shaders/scene.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	if (!vertex_module || !fragment_module)
		goto cleanup;

	spudgpu_shader_pipeline_desc pipeline_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SHADER_PIPELINE_DESC, NULL},
	    .vertex_module   = vertex_module,
	    .fragment_module = fragment_module,
	    .vertex_attributes = {
	        {.location = 0, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, position)},
	        {.location = 1, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, normal)},
	    },
	    .vertex_attribute_count = 2,
	    .vertex_bindings        = {{.binding = 0, .stride = sizeof(Vertex), .per_instance = false}},
	    .vertex_binding_count   = 1,
	    .primitive_topology     = SPUDGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	    .cull_mode              = SPUDGPU_CULL_MODE_NONE,
	    .front_face_ccw         = true,
	    .depth_test_enable      = true,
	    .depth_write_enable     = true,
	    .depth_compare_op       = SPUDGPU_COMPARE_OP_LESS,
	    .color_attachment_formats = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count  = 1,
	    .depth_format            = SPUDGPU_FORMAT_D32_FLOAT,
	    .descriptor_set_layouts      = {cbv_layout},
	    .descriptor_set_layout_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		goto cleanup;
	}

	spudgpu_image_desc depth_image_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_IMAGE_DESC, NULL},
	    .usage        = SPUDGPU_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT,
	    .type         = SPUDGPU_IMAGE_TYPE_2D,
	    .format       = SPUDGPU_FORMAT_D32_FLOAT,
	    .width        = WINDOW_WIDTH,
	    .height       = WINDOW_HEIGHT,
	    .depth        = 1,
	    .array_layers = 1,
	    .mip_levels   = 1,
	    .clear_value  = {.format = SPUDGPU_FORMAT_D32_FLOAT, .depth_stencil = {.depth = 1.0f, .stencil = 0}},
	};
	if (SPUDFAIL(spudgpu_create_image(device, &depth_image_desc, &depth_image))) {
		fprintf(stderr, "spudgpu_create_image (depth) failed\n");
		goto cleanup;
	}
	spudgpu_image_view_desc depth_view_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_IMAGE_VIEW_DESC, NULL},
	    .parent_image = depth_image,
	    .type         = SPUDGPU_IMAGE_VIEW_TYPE_2D,
	    .subresource_range = {.aspect_mask = SPUDGPU_ASPECT_DEPTH_BIT, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
	};
	spudgpu_create_image_view(depth_image, &depth_view_desc, &depth_view);

	// cmd was created via spudgpu_create_command_list but never begun -- a
	// freshly created command list starts CLOSED, so recording this barrier
	// needs its own begin/end/submit bracket rather than assuming cmd is
	// already open (the render loop's first spudgpu_begin_command_list call
	// happens later and would otherwise discard this on Reset).
	spudgpu_begin_command_list(cmd);
	spudgpu_cmd_image_barrier(cmd, depth_image, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);

	// ------------------------------------------------------------------
	// Bundle: record the entire per-object bind+draw sequence once, up
	// front. Never re-recorded -- see the file comment above.
	// ------------------------------------------------------------------

	spudgpu_command_allocator_desc bundle_allocator_desc = {.base = {SPUD_STRUCT_TYPE_GPU_COMMAND_ALLOCATOR_DESC, NULL}, .type = SPUDGPU_COMMAND_LIST_TYPE_BUNDLE};
	if (SPUDFAIL(spudgpu_create_command_allocator(device, &bundle_allocator_desc, &bundle_allocator))) {
		fprintf(stderr, "spudgpu_create_command_allocator (bundle) failed\n");
		goto cleanup;
	}
	if (SPUDFAIL(spudgpu_create_command_list(bundle_allocator, &bundle))) {
		fprintf(stderr, "spudgpu_create_command_list (bundle) failed\n");
		goto cleanup;
	}

	spudgpu_bundle_inheritance_desc bundle_inheritance = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUNDLE_INHERITANCE_DESC, NULL},
	    .color_attachment_formats = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count  = 1,
	    .depth_format            = SPUDGPU_FORMAT_D32_FLOAT,
	    .viewport                = window_viewport,
	    .scissor_rect            = window_scissor,
	};
	spudgpu_begin_bundle_command_list(bundle, &bundle_inheritance);
	record_city_draws(bundle, pipeline, vertex_buffer_view, index_buffer_view, index_count, object_sets);
	spudgpu_end_command_list(bundle);

	// ------------------------------------------------------------------
	// Main loop.
	// ------------------------------------------------------------------

	Camera camera;
	camera_init(&camera, ((CITY_COLUMN_COUNT - 1) / 2.0f) * CITY_SPACING_INTERVAL, 8.0f, 30.0f, CITY_SPACING_INTERVAL * 2.0f);

	const float up[3] = {0.0f, 1.0f, 0.0f};
	uint64_t last_ticks = SDL_GetTicks();

	bool running     = true;
	bool use_bundles = true;

	while (running) {
		if (frame_limit != 0 && frames_run == frame_limit)
			break;
		++frames_run;
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
			if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_C && !event.key.repeat) {
				use_bundles = !use_bundles;
				SDL_SetWindowTitle(window, use_bundles ? "SpudGPU Bundles [Bundles ON]" : "SpudGPU Bundles [Bundles OFF - recording every frame]");
				printf("Bundles %s\n", use_bundles ? "ON (recorded once, replayed)" : "OFF (re-recorded every frame)");
			}
			if (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) {
				bool down = event.type == SDL_EVENT_KEY_DOWN;
				switch (event.key.scancode) {
				case SDL_SCANCODE_W: camera.key_w = down; break;
				case SDL_SCANCODE_A: camera.key_a = down; break;
				case SDL_SCANCODE_S: camera.key_s = down; break;
				case SDL_SCANCODE_D: camera.key_d = down; break;
				case SDL_SCANCODE_LEFT: camera.key_left = down; break;
				case SDL_SCANCODE_RIGHT: camera.key_right = down; break;
				case SDL_SCANCODE_UP: camera.key_up = down; break;
				case SDL_SCANCODE_DOWN: camera.key_down = down; break;
				case SDL_SCANCODE_ESCAPE:
					if (down) running = false;
					break;
				default: break;
				}
			}
		}

		uint64_t now = SDL_GetTicks();
		float dt     = (float) (now - last_ticks) / 1000.0f;
		last_ticks   = now;
		camera_update(&camera, dt);

		float view[16], proj[16], view_proj[16];
		mat4_look_to_rh(camera.position, camera.look_dir, up, view);
		mat4_perspective_rh(0.8f, (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT, 1.0f, 1000.0f, proj);
		mat4_multiply(proj, view, view_proj);

		for (uint32_t i = 0; i < CITY_OBJECT_COUNT; i++) {
			float mvp[16];
			mat4_multiply(view_proj, model_matrices[i], mvp);
			memcpy((uint8_t *) cbv_mapped + i * cbv_stride, mvp, sizeof(mvp));
			spudgpu_flush_buffer(cbv_buffer, i * cbv_stride, sizeof(mvp));
		}

		uint32_t image_index               = spudgpu_swap_chain_acquire_next_image(swap_chain);
		spudgpu_image_view backbuffer_view = spudgpu_get_swap_chain_image_view(swap_chain, image_index);

		spudgpu_reset_command_allocator(command_allocator);
		spudgpu_begin_command_list(cmd);

		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		spudgpu_rendering_begin_desc rendering_desc = {
		    .base = {SPUD_STRUCT_TYPE_GPU_RENDERING_BEGIN_DESC, NULL},
		    .color_attachments = {{.image_view = backbuffer_view, .load_op = SPUDGPU_LOAD_OP_CLEAR, .store_op = SPUDGPU_STORE_OP_STORE, .clear_color = {0.0f, 0.2f, 0.4f, 1.0f}}},
		    .color_attachment_count = 1,
		    .depth_attachment       = {.image_view = depth_view, .depth_load_op = SPUDGPU_LOAD_OP_CLEAR, .depth_store_op = SPUDGPU_STORE_OP_STORE, .clear_depth = 1.0f},
		    .width                  = WINDOW_WIDTH,
		    .height                 = WINDOW_HEIGHT,
		    .will_execute_bundles   = use_bundles,
		};
		spudgpu_cmd_begin_rendering(cmd, &rendering_desc);

		// Viewport/scissor are set here, on the direct list, every frame --
		// a bundle can't set its own (see record_city_draws above). They are
		// the same two the bundle was given when it was recorded.
		spudgpu_cmd_set_viewports(cmd, 0, 1, &window_viewport);
		spudgpu_cmd_set_scissor_rects(cmd, 0, 1, &window_scissor);

		if (use_bundles) {
			// ExecuteBundle requires the direct list's descriptor heaps to
			// already match the bundle's -- bind the pipeline (root
			// signature, needed before any SetGraphicsRootDescriptorTable
			// call) and one object set on cmd purely to prime the heap; the
			// bundle's own per-building binds (same pool, same heap)
			// immediately overwrite the root table this sets for object 0.
			spudgpu_cmd_bind_pipeline(cmd, pipeline);
			spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 0, &object_sets[0], 1);
			spudgpu_cmd_execute_bundle(cmd, bundle);
		} else {
			record_city_draws(cmd, pipeline, vertex_buffer_view, index_buffer_view, index_count, object_sets);
		}

		spudgpu_cmd_end_rendering(cmd);
		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, SPUDGPU_IMAGE_LAYOUT_PRESENT_SRC);

		spudgpu_end_command_list(cmd);
		spudgpu_submit_command_lists_synced(graphics_queue, &cmd, 1, swap_chain);
		spudgpu_swap_chain_present(swap_chain);
	}

	exit_code = 0;

cleanup:
	if (graphics_queue)
		spudgpu_queue_wait_idle(graphics_queue);

	spudgpu_destroy_command_list(bundle);
	spudgpu_destroy_command_allocator(bundle_allocator);
	spudgpu_destroy_image_view(depth_view);
	spudgpu_destroy_image(depth_image);
	spudgpu_destroy_shader_pipeline(pipeline);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(vertex_module);
	if (cbv_buffer)
		spudgpu_unmap_buffer(cbv_buffer);
	spudgpu_destroy_buffer(cbv_buffer);
	spudgpu_destroy_descriptor_pool(cbv_pool);
	spudgpu_destroy_descriptor_set_layout(cbv_layout);
	spudgpu_destroy_buffer_view(index_buffer_view);
	spudgpu_destroy_buffer(index_buffer);
	spudgpu_destroy_buffer_view(vertex_buffer_view);
	spudgpu_destroy_buffer(vertex_buffer);
	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_swap_chain(swap_chain);
	spudgpu_destroy_surface(surface);
	spudgpu_destroy_device(device);
	spudgpu_destroy_instance(instance);

	if (window)
		SDL_DestroyWindow(window);
	SDL_Quit();
	return exit_code;
}
