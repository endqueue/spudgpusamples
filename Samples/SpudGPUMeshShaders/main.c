//
// SpudGPU port of D3D12MeshShaders/MeshletRender from d3d12samples -- the
// simplest of that sample family (no amplification/task shader, no meshlet
// culling, no textures). Renders a pre-built meshlet binary (Dragon_LOD0.bin,
// copied verbatim from the original sample, no converter tool needed) via
// SpudGPU's mesh-shader pipeline, first exercised here -- see ../../README.md
// for the one deliberate simplification versus the original.
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

// ---------------------------------------------------------------------------
// Dragon_LOD0.bin file format -- mirrors d3d12samples' Model.h/Model.cpp
// exactly (FileHeader | MeshHeader[] | Accessor[] | BufferView[] | raw blob,
// with MeshHeader fields being accessor indices, each accessor pointing at a
// buffer-view index). Verified against the actual file bytes (magic
// 0x4D53484C == 'MSHL', header sizes/offsets add up to the real file size)
// before writing this.
// ---------------------------------------------------------------------------

#define MESHLET_FILE_PROLOG 0x4D53484Cu // 'MSHL'
#define MESHLET_ATTRIBUTE_POSITION 0u
#define MESHLET_ATTRIBUTE_NORMAL 1u
#define MESHLET_INVALID_ACCESSOR 0xFFFFFFFFu

typedef struct FileHeader {
	uint32_t prolog;
	uint32_t version;
	uint32_t mesh_count;
	uint32_t accessor_count;
	uint32_t buffer_view_count;
	uint32_t buffer_size;
} FileHeader;

typedef struct FileMeshHeader {
	uint32_t indices;
	uint32_t index_subsets;
	uint32_t attributes[5]; // Position, Normal, TexCoord, Tangent, Bitangent
	uint32_t meshlets;
	uint32_t meshlet_subsets;
	uint32_t unique_vertex_indices;
	uint32_t primitive_indices;
	uint32_t cull_data;
} FileMeshHeader;

typedef struct FileAccessor {
	uint32_t buffer_view;
	uint32_t offset;
	uint32_t size;
	uint32_t stride;
	uint32_t count;
} FileAccessor;

typedef struct FileBufferView {
	uint32_t offset;
	uint32_t size;
} FileBufferView;

// GPU-side structs -- must match shaders/mesh.mesh's Vertex/Meshlet/Subset
// layouts exactly (std430 storage-buffer layout: all-uint32/vec3 members,
// naturally 4-byte-aligned, no manual padding needed).
typedef struct Vertex {
	float position[3];
	float normal[3];
} Vertex; // stride 24, matches the file's interleaved Position+Normal buffer view.

typedef struct Meshlet {
	uint32_t vert_count;
	uint32_t vert_offset;
	uint32_t prim_count;
	uint32_t prim_offset;
} Meshlet;

typedef struct Subset {
	uint32_t offset;
	uint32_t count;
} Subset;

typedef struct PushConstants {
	uint32_t index_bytes;
	uint32_t meshlet_offset;
} PushConstants;

typedef struct Globals {
	float world[16];
	float world_view[16];
	float world_view_proj[16];
	uint32_t draw_meshlets;
	float _pad[3]; // round up to a multiple of 16 bytes for std140/CBV alignment.
} Globals;

// Resolves a MeshHeader field (an accessor index, or MESHLET_INVALID_ACCESSOR
// if unused) to the raw byte range in the blob that its buffer view covers.
// Every resource this sample loads is either its own dedicated buffer view
// (Meshlets/MeshletSubsets/UniqueVertexIndices/PrimitiveIndices) or, for
// Position/Normal, the one interleaved buffer view both attributes share --
// either way the whole buffer-view range is what we want to upload verbatim,
// not any single attribute's own accessor sub-window within it.
static bool resolve_buffer_view(
    const FileAccessor *accessors,
    const FileBufferView *buffer_views,
    uint32_t accessor_index,
    uint32_t *out_offset,
    uint32_t *out_size) {
	if (accessor_index == MESHLET_INVALID_ACCESSOR)
		return false;
	uint32_t bv_index = accessors[accessor_index].buffer_view;
	*out_offset        = buffer_views[bv_index].offset;
	*out_size          = buffer_views[bv_index].size;
	return true;
}

// ---------------------------------------------------------------------------
// Math (column-major, right-handed) -- identical to SpudGPUDynamicIndexing's,
// which already ported this exact model/view/projection convention.
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

static void mat4_identity(float out[16]) {
	memset(out, 0, sizeof(float) * 16);
	out[0] = out[5] = out[10] = out[15] = 1.0f;
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
// Camera -- identical keyboard-only WASD+arrows camera already ported twice
// (SpudGPUExecuteIndirect, SpudGPUDynamicIndexing); this sample's own move
// speed/FOV/initial position match the original MeshletRender's SimpleCamera
// overrides.
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

// ---------------------------------------------------------------------------

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

// Creates a storage buffer in DEVICE memory and uploads [data, data+size)
// into it once, via a temporary UPLOAD staging buffer + spudgpu_cmd_copy_buffer.
// SPUDGPU_MEMORY_KIND_UPLOAD refuses SPUDGPU_BUFFER_USAGE_STORAGE on every
// backend, and the one mappable kind that takes it
// (SPUDGPU_MEMORY_KIND_DEVICE_MAPPABLE) isn't on every device -- so this
// can't just map the real buffer directly the way every other one-time
// upload in this suite does. Synchronous (one begin/end/submit/wait per call): this only
// runs a handful of times at startup, so the simplicity is worth more than
// batching all four uploads into one submission. NULL, printed, on failure.
static spudgpu_buffer create_storage_buffer_with_data(
    spudgpu_device device, spudgpu_command_list cmd, spudgpu_command_queue graphics_queue,
    const void *data, uint64_t size) {
	spudgpu_buffer_desc staging_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_TRANSFER_SRC,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = size,
	};
	spudgpu_buffer staging = NULL;
	if (SPUDFAIL(spudgpu_create_buffer(device, &staging_desc, &staging))) {
		fprintf(stderr, "spudgpu_create_buffer (staging) failed\n");
		return NULL;
	}
	void *mapped = NULL;
	spudgpu_map_buffer(staging, 0, 0, &mapped);
	memcpy(mapped, data, size);
	spudgpu_flush_buffer(staging, 0, 0);
	spudgpu_unmap_buffer(staging);

	spudgpu_buffer_desc real_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_STORAGE | SPUDGPU_BUFFER_USAGE_TRANSFER_DST,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_DEVICE,
	    .size         = size,
	};
	spudgpu_buffer buffer = NULL;
	if (SPUDFAIL(spudgpu_create_buffer(device, &real_desc, &buffer))) {
		fprintf(stderr, "spudgpu_create_buffer (storage) failed\n");
		spudgpu_destroy_buffer(staging);
		return NULL;
	}

	// No COMMON -> UNORDERED_ACCESS barrier here: D3D12 implicitly promotes
	// this buffer to COPY_DEST for the copy itself, and that promoted state
	// doesn't decay back to COMMON until this command list actually finishes
	// executing (at the ExecuteCommandLists boundary) -- a barrier recorded
	// right here, in the same not-yet-submitted command list, would have to
	// declare "before = COPY_DEST" to match, not COMMON. Simplest correct
	// fix: transition to UNORDERED_ACCESS later, in a separate command list
	// recorded after this one has been submitted and the queue drained
	// (see the buffer_barriers passed alongside the depth image's barrier
	// below) -- by then the buffer has legitimately decayed back to COMMON.
	spudgpu_begin_command_list(cmd);
	spudgpu_cmd_copy_buffer(cmd, staging, buffer, 0, 0, size);
	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);

	spudgpu_destroy_buffer(staging);
	return buffer;
}

int main(int argc, char **argv) {
	// --frames N: render N frames, then leave the loop as closing the window
	// does, through the same cleanup. Without it the sample runs until the
	// window is closed.
	// --device N: use the Nth device spudgpu_enumerate_devices lists. 0 if
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
	int exit_code                                  = 1;
	SDL_Window *window                             = NULL;
	spudgpu_instance instance                      = NULL;
	spudgpu_command_queue graphics_queue           = NULL;
	spudgpu_surface surface                        = NULL;
	spudgpu_swap_chain swap_chain                  = NULL;
	spudgpu_command_allocator command_allocator    = NULL;
	spudgpu_command_list cmd                       = NULL;
	uint8_t *file_data                             = NULL;
	Subset *meshlet_subsets                        = NULL;
	spudgpu_buffer vertices_buffer                 = NULL;
	spudgpu_buffer meshlets_buffer                 = NULL;
	spudgpu_buffer unique_vtx_idx_buffer           = NULL;
	spudgpu_buffer primitive_idx_buffer            = NULL;
	spudgpu_descriptor_set_layout mesh_data_layout = NULL;
	spudgpu_descriptor_pool mesh_data_pool         = NULL;
	spudgpu_descriptor_set_layout globals_layout   = NULL;
	spudgpu_descriptor_pool globals_pool           = NULL;
	spudgpu_buffer globals_buffer                  = NULL;
	spudgpu_shader_module mesh_module              = NULL;
	spudgpu_shader_module fragment_module          = NULL;
	spudgpu_shader_pipeline pipeline               = NULL;
	spudgpu_image depth_image                      = NULL;
	spudgpu_image_view depth_view                  = NULL;

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	window = SDL_CreateWindow("SpudGPU MeshShaders", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		goto cleanup;
	}

	if (SPUDFAIL(spudgpu_create_instance("SpudGPUMeshShaders", 1, "SpudGPUSamples", 1, &instance))) {
		fprintf(stderr, "spudgpu_create_instance failed\n");
		goto cleanup;
	}

	spudgpu_device *devices = NULL;
	uint32_t device_count   = 0;
	if (SPUDFAIL(spudgpu_enumerate_devices(instance, &devices, &device_count)) || device_count == 0) {
		fprintf(stderr, "spudgpu_enumerate_devices failed or returned no devices\n");
		goto cleanup;
	}
	if (device_index >= device_count) {
		fprintf(stderr, "--device %u: there are %u devices\n", device_index, device_count);
		goto cleanup;
	}
	spudgpu_device device = devices[device_index];

	SPUDGPU_DEVICE_PROPERTIES device_properties = {.base = {SPUD_STRUCT_TYPE_GPU_DEVICE_PROPERTIES, NULL}};
	spudgpu_get_device_properties(device, &device_properties);
	printf("Using device: %s\n", device_properties.description);

	spudgpu_mesh_shading_capabilities mesh_caps = {.base = {SPUD_STRUCT_TYPE_GPU_MESH_SHADING_CAPABILITIES, NULL}};
	spudgpu_get_mesh_shading_capabilities(device, &mesh_caps);
	if (!mesh_caps.supported) {
		fprintf(stderr, "This device/driver doesn't support mesh shading\n");
		goto cleanup;
	}

	graphics_queue = spudgpu_get_graphics_queue(device);

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

	// Created here (rather than just before the main loop, as in most other
	// samples) because create_storage_buffer_with_data below already needs a
	// command list to record its staging-buffer upload into.
	spudgpu_command_allocator_desc allocator_desc = {.base = {SPUD_STRUCT_TYPE_GPU_COMMAND_ALLOCATOR_DESC, NULL}, .type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_create_command_list(command_allocator, &cmd);

	// ------------------------------------------------------------------
	// Load and parse Dragon_LOD0.bin (single mesh).
	// ------------------------------------------------------------------

	size_t file_size;
	file_data = (uint8_t *) read_entire_file("Dragon_LOD0.bin", &file_size);
	if (!file_data)
		goto cleanup;

	const FileHeader *header = (const FileHeader *) file_data;
	if (header->prolog != MESHLET_FILE_PROLOG || header->mesh_count == 0) {
		fprintf(stderr, "Dragon_LOD0.bin: bad file header\n");
		goto cleanup;
	}
	const FileMeshHeader *meshes = (const FileMeshHeader *) (file_data + sizeof(FileHeader));
	const FileAccessor *accessors = (const FileAccessor *) ((const uint8_t *) meshes + (size_t) header->mesh_count * sizeof(FileMeshHeader));
	const FileBufferView *buffer_views = (const FileBufferView *) ((const uint8_t *) accessors + (size_t) header->accessor_count * sizeof(FileAccessor));
	const uint8_t *blob = (const uint8_t *) buffer_views + (size_t) header->buffer_view_count * sizeof(FileBufferView);

	const FileMeshHeader *mesh = &meshes[0];

	uint32_t vtx_offset, vtx_size;
	uint32_t meshlet_offset, meshlet_size;
	uint32_t meshlet_subset_offset, meshlet_subset_size;
	uint32_t uvi_offset, uvi_size;
	uint32_t pi_offset, pi_size;
	if (!resolve_buffer_view(accessors, buffer_views, mesh->attributes[MESHLET_ATTRIBUTE_POSITION], &vtx_offset, &vtx_size) ||
	    !resolve_buffer_view(accessors, buffer_views, mesh->meshlets, &meshlet_offset, &meshlet_size) ||
	    !resolve_buffer_view(accessors, buffer_views, mesh->meshlet_subsets, &meshlet_subset_offset, &meshlet_subset_size) ||
	    !resolve_buffer_view(accessors, buffer_views, mesh->unique_vertex_indices, &uvi_offset, &uvi_size) ||
	    !resolve_buffer_view(accessors, buffer_views, mesh->primitive_indices, &pi_offset, &pi_size)) {
		fprintf(stderr, "Dragon_LOD0.bin: missing a required buffer view\n");
		goto cleanup;
	}
	// Only the Indices accessor's Size is needed (2 or 4 bytes per index --
	// see the original's Model.cpp: "mesh.IndexSize = accessor.Size") -- the
	// actual index buffer it describes is unused by this sample's mesh
	// shader, same as the original. Size, not Stride: for a tightly packed
	// scalar index buffer the two happen to differ in this asset (Stride
	// here is 0, since nothing else shares this accessor's buffer view),
	// so reading Stride was silently feeding get_vertex_index() the wrong
	// index_bytes and corrupting every 16-bit-packed vertex-index lookup.
	uint32_t index_bytes = accessors[mesh->indices].size;

	// Copied out of the file: the render loop reads them every frame, and
	// file_data is freed below. They once pointed into it, and the loop read
	// freed memory.
	uint32_t meshlet_subset_count = meshlet_subset_size / sizeof(Subset);
	meshlet_subsets               = (Subset *) malloc(meshlet_subset_size);
	if (!meshlet_subsets) {
		fprintf(stderr, "Out of memory copying the meshlet subsets\n");
		goto cleanup;
	}
	memcpy(meshlet_subsets, blob + meshlet_subset_offset, meshlet_subset_size);

	vertices_buffer       = create_storage_buffer_with_data(device, cmd, graphics_queue, blob + vtx_offset, vtx_size);
	meshlets_buffer       = create_storage_buffer_with_data(device, cmd, graphics_queue, blob + meshlet_offset, meshlet_size);
	unique_vtx_idx_buffer = create_storage_buffer_with_data(device, cmd, graphics_queue, blob + uvi_offset, uvi_size);
	primitive_idx_buffer  = create_storage_buffer_with_data(device, cmd, graphics_queue, blob + pi_offset, pi_size);
	if (!vertices_buffer || !meshlets_buffer || !unique_vtx_idx_buffer || !primitive_idx_buffer)
		goto cleanup;

	free(file_data);
	file_data = NULL;

	// ------------------------------------------------------------------
	// Descriptor set 0: the 4 mesh-data storage buffers (mesh-visible).
	// ------------------------------------------------------------------

	spudgpu_descriptor_set_layout_desc mesh_data_layout_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_LAYOUT_DESC, NULL},
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_MESH},
	        {.binding = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_MESH},
	        {.binding = 2, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_MESH},
	        {.binding = 3, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_MESH},
	    },
	    .binding_count = 4,
	};
	spudgpu_create_descriptor_set_layout(device, &mesh_data_layout_desc, &mesh_data_layout);

	spudgpu_descriptor_pool_desc mesh_data_pool_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_POOL_DESC, NULL},
	    .max_sets = 1, .pool_sizes = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 4}}, .pool_size_count = 1};
	spudgpu_create_descriptor_pool(device, &mesh_data_pool_desc, &mesh_data_pool);

	spudgpu_descriptor_set_desc mesh_data_set_desc = {.base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_DESC, NULL}, .pool = mesh_data_pool, .set_layouts = {mesh_data_layout}, .set_count = 1};
	spudgpu_descriptor_set mesh_data_set = NULL;
	spudgpu_create_descriptor_sets(device, &mesh_data_set_desc, &mesh_data_set);

	spudgpu_descriptor_buffer_info vertices_info = {.buffer = vertices_buffer, .range = vtx_size};
	spudgpu_descriptor_buffer_info meshlets_info = {.buffer = meshlets_buffer, .range = meshlet_size};
	spudgpu_descriptor_buffer_info uvi_info      = {.buffer = unique_vtx_idx_buffer, .range = uvi_size};
	spudgpu_descriptor_buffer_info pi_info       = {.buffer = primitive_idx_buffer, .range = pi_size};
	spudgpu_write_descriptor_set mesh_data_writes[4] = {
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = mesh_data_set, .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = &vertices_info},
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = mesh_data_set, .dst_binding = 1, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = &meshlets_info},
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = mesh_data_set, .dst_binding = 2, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = &uvi_info},
	    {.base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL}, .dst_set = mesh_data_set, .dst_binding = 3, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = &pi_info},
	};
	spudgpu_update_descriptor_sets(device, mesh_data_writes, 4);

	// ------------------------------------------------------------------
	// Descriptor set 1: Globals CBV (mesh + fragment visible).
	// ------------------------------------------------------------------

	spudgpu_descriptor_set_layout_desc globals_layout_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_LAYOUT_DESC, NULL},
	    .bindings = {{.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_MESH | SPUDGPU_SHADER_STAGE_FRAGMENT}},
	    .binding_count = 1,
	};
	spudgpu_create_descriptor_set_layout(device, &globals_layout_desc, &globals_layout);

	spudgpu_descriptor_pool_desc globals_pool_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_POOL_DESC, NULL},
	    .max_sets = 1, .pool_sizes = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1}}, .pool_size_count = 1};
	spudgpu_create_descriptor_pool(device, &globals_pool_desc, &globals_pool);

	spudgpu_descriptor_set_desc globals_set_desc = {.base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_DESC, NULL}, .pool = globals_pool, .set_layouts = {globals_layout}, .set_count = 1};
	spudgpu_descriptor_set globals_set = NULL;
	spudgpu_create_descriptor_sets(device, &globals_set_desc, &globals_set);

	spudgpu_buffer_desc globals_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(Globals),
	};
	spudgpu_create_buffer(device, &globals_buffer_desc, &globals_buffer);
	void *globals_mapped = NULL;
	spudgpu_map_buffer(globals_buffer, 0, 0, &globals_mapped);

	spudgpu_descriptor_buffer_info globals_info = {.buffer = globals_buffer, .range = sizeof(Globals)};
	spudgpu_write_descriptor_set globals_write = {
	    .base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL},
	    .dst_set = globals_set, .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer_info = &globals_info};
	spudgpu_update_descriptor_sets(device, &globals_write, 1);

	// ------------------------------------------------------------------
	// Mesh-shader pipeline + depth buffer.
	// ------------------------------------------------------------------

	mesh_module     = load_shader_module(device, "shaders/mesh.mesh.spv", SPUDGPU_SHADER_STAGE_MESH);
	fragment_module = load_shader_module(device, "shaders/scene.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	if (!mesh_module || !fragment_module)
		goto cleanup;

	spudgpu_shader_pipeline_desc pipeline_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SHADER_PIPELINE_DESC, NULL},
	    .mesh_module     = mesh_module,
	    .fragment_module = fragment_module,
	    .primitive_topology = SPUDGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	    .cull_mode          = SPUDGPU_CULL_MODE_BACK,
	    .front_face_ccw     = false, // matches D3D12_DEFAULT's clockwise-is-front-facing.
	    .depth_test_enable  = true,
	    .depth_write_enable = true,
	    .depth_compare_op   = SPUDGPU_COMPARE_OP_LESS,
	    .color_attachment_formats = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count  = 1,
	    .depth_format            = SPUDGPU_FORMAT_D32_FLOAT,
	    .descriptor_set_layouts      = {mesh_data_layout, globals_layout},
	    .descriptor_set_layout_count = 2,
	    .push_constant_ranges        = {{.offset = 0, .size = sizeof(PushConstants), .stage_flags = SPUDGPU_SHADER_STAGE_MESH}},
	    .push_constant_range_count   = 1,
	};
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		goto cleanup;
	}

	// depth image created/transitioned via `cmd` before the main loop starts.
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
	    .subresource_range = {.aspect_mask = 0x2 /* DEPTH, Vulkan-aligned - see spudgpu.h */, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
	};
	spudgpu_create_image_view(depth_image, &depth_view_desc, &depth_view);

	spudgpu_begin_command_list(cmd);
	spudgpu_cmd_image_barrier(cmd, depth_image, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	// Mesh shaders read these as UAVs (SPUDGPU_DESCRIPTOR_TYPE_STORAGE_BUFFER
	// maps to D3D12_DESCRIPTOR_RANGE_TYPE_UAV) -- unlike the read-only states
	// D3D12 promotes to implicitly, UNORDERED_ACCESS needs an explicit
	// transition. COMMON is valid as "before" here specifically because each
	// buffer's own upload command list (create_storage_buffer_with_data) has
	// already been submitted and the queue drained by this point, so the
	// COPY_DEST state that upload promoted to has already decayed back to
	// COMMON -- see that function's comment for why the same transition
	// can't happen inside its own command list.
	spudgpu_buffer_barrier buffer_barriers[4] = {
	    {.base = {SPUD_STRUCT_TYPE_GPU_BUFFER_BARRIER, NULL}, .buffer = vertices_buffer, .state_before = SPUDGPU_RESOURCE_STATE_COMMON, .state_after = SPUDGPU_RESOURCE_STATE_UNORDERED_ACCESS},
	    {.base = {SPUD_STRUCT_TYPE_GPU_BUFFER_BARRIER, NULL}, .buffer = meshlets_buffer, .state_before = SPUDGPU_RESOURCE_STATE_COMMON, .state_after = SPUDGPU_RESOURCE_STATE_UNORDERED_ACCESS},
	    {.base = {SPUD_STRUCT_TYPE_GPU_BUFFER_BARRIER, NULL}, .buffer = unique_vtx_idx_buffer, .state_before = SPUDGPU_RESOURCE_STATE_COMMON, .state_after = SPUDGPU_RESOURCE_STATE_UNORDERED_ACCESS},
	    {.base = {SPUD_STRUCT_TYPE_GPU_BUFFER_BARRIER, NULL}, .buffer = primitive_idx_buffer, .state_before = SPUDGPU_RESOURCE_STATE_COMMON, .state_after = SPUDGPU_RESOURCE_STATE_UNORDERED_ACCESS},
	};
	spudgpu_cmd_pipeline_barrier(cmd, buffer_barriers, 4, NULL, 0);
	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);

	// ------------------------------------------------------------------
	// Main loop.
	// ------------------------------------------------------------------

	Camera camera;
	camera_init(&camera, 0.0f, 75.0f, 150.0f, 150.0f);

	const float up[3] = {0.0f, 1.0f, 0.0f};
	uint64_t last_ticks = SDL_GetTicks();

	bool running = true;
	while (running) {
		if (frame_limit != 0 && frames_run == frame_limit)
			break;
		++frames_run;
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
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

		Globals globals_data;
		mat4_identity(globals_data.world);
		float view[16], proj[16];
		mat4_look_to_rh(camera.position, camera.look_dir, up, view);
		mat4_perspective_rh((float) M_PI / 3.0f, (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT, 1.0f, 1000.0f, proj);
		memcpy(globals_data.world_view, view, sizeof(view)); // world is identity.
		mat4_multiply(proj, view, globals_data.world_view_proj);
		globals_data.draw_meshlets = 1;
		memcpy(globals_mapped, &globals_data, sizeof(globals_data));
		spudgpu_flush_buffer(globals_buffer, 0, sizeof(globals_data));

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
		};
		spudgpu_cmd_begin_rendering(cmd, &rendering_desc);

		spudgpu_cmd_bind_pipeline(cmd, pipeline);
		spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 0, &mesh_data_set, 1);
		spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 1, &globals_set, 1);

		SPUDGPU_VIEWPORT viewport = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT, .minDepth = 0.0f, .maxDepth = 1.0f};
		spudgpu_cmd_set_viewports(cmd, 0, 1, &viewport);
		SPUDGPU_SCISSOR_RECT scissor = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};
		spudgpu_cmd_set_scissor_rects(cmd, 0, 1, &scissor);

		for (uint32_t s = 0; s < meshlet_subset_count; s++) {
			PushConstants pc = {.index_bytes = index_bytes, .meshlet_offset = meshlet_subsets[s].offset};
			spudgpu_cmd_push_constants(cmd, pipeline, 0, sizeof(pc), &pc);
			spudgpu_cmd_dispatch_mesh(cmd, meshlet_subsets[s].count, 1, 1);
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

	spudgpu_destroy_image_view(depth_view);
	spudgpu_destroy_image(depth_image);
	spudgpu_destroy_shader_pipeline(pipeline);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(mesh_module);
	free(meshlet_subsets);
	free(file_data);
	if (globals_buffer)
		spudgpu_unmap_buffer(globals_buffer);
	spudgpu_destroy_buffer(globals_buffer);
	spudgpu_destroy_descriptor_pool(globals_pool);
	spudgpu_destroy_descriptor_set_layout(globals_layout);
	spudgpu_destroy_descriptor_pool(mesh_data_pool);
	spudgpu_destroy_descriptor_set_layout(mesh_data_layout);
	spudgpu_destroy_buffer(primitive_idx_buffer);
	spudgpu_destroy_buffer(unique_vtx_idx_buffer);
	spudgpu_destroy_buffer(meshlets_buffer);
	spudgpu_destroy_buffer(vertices_buffer);
	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_swap_chain(swap_chain);
	spudgpu_destroy_surface(surface);
	spudgpu_destroy_instance(instance);

	if (window)
		SDL_DestroyWindow(window);
	SDL_Quit();
	return exit_code;
}
