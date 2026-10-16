//
// SpudGPU port of D3D12DynamicIndexing from d3d12samples. A 15x8 grid of
// identical "occcity" building meshes; each building's pixel shader
// dynamically indexes into one shared unbounded texture table (SpudGPU's
// bindless sampled-image feature) to pick its own procedurally-generated
// material color, blended against a shared diffuse texture -- see
// ../../README.md for the two things this port does differently from the
// original and why.
//
// occcity.bin is copied verbatim from
// d3d12samples/Samples/Desktop/D3D12DynamicIndexing/src/ -- the fixed byte
// offsets/sizes below (VERTEX_DATA_OFFSET etc.) come directly from that
// sample's occcity.h.
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

#define CITY_ROW_COUNT 15
#define CITY_COLUMN_COUNT 8
#define CITY_MATERIAL_COUNT (CITY_ROW_COUNT * CITY_COLUMN_COUNT) // 120
#define CITY_MATERIAL_TEXTURE_WIDTH 64
#define CITY_MATERIAL_TEXTURE_HEIGHT 64
#define CITY_SPACING_INTERVAL 16.0f

// occcity.bin layout, from d3d12samples' occcity.h (SampleAssets namespace).
#define VERTEX_DATA_OFFSET 524288u
#define VERTEX_DATA_SIZE 820248u
#define INDEX_DATA_OFFSET 1344536u
#define INDEX_DATA_SIZE 74568u
#define VERTEX_STRIDE 44u
#define DIFFUSE_BC1_OFFSET 0u
#define DIFFUSE_BC1_ROW_PITCH 2048u
#define DIFFUSE_TEXTURE_WIDTH 1024u
#define DIFFUSE_TEXTURE_HEIGHT 1024u

// Vulkan-aligned aspect bits (COLOR=1, DEPTH=2, STENCIL=4) -- spudgpu.h's
// spudgpu_image_view_desc_subresource_range::aspect_mask has no named
// constants yet; the Vulkan backend passes this straight through to
// VkImageAspectFlags (see spudgpuvulkanimage.c), and D3D12/Metal ignore it
// entirely (they infer DSV/RTV/SRV purely from the image's usage bits).
#define SPUDGPU_ASPECT_DEPTH_BIT 0x00000002u

typedef struct Vertex {
	float position[3];
	float normal[3]; // Unused by these shaders (no lighting), kept only so
	                  // the vertex stride/offsets match occcity.bin's layout.
	float uv[2];
	float tangent[3]; // Unused.
} Vertex;

typedef struct PushConstants {
	uint32_t diffuse_index;
	uint32_t material_index;
} PushConstants;

// ---------------------------------------------------------------------------
// Math (column-major, right-handed -- matches SimpleCamera's
// XMMatrixLookToRH/PerspectiveFovRH exactly, unlike SpudGPUExecuteIndirect's
// left-handed projection).
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
// BC1 (DXT1) -> RGBA8 CPU decode. See ../../README.md for why this port
// decodes on load instead of uploading the compressed texture directly.
// ---------------------------------------------------------------------------

static void bc1_unpack_565(uint16_t c, uint8_t rgb[3]) {
	rgb[0] = (uint8_t)(((c >> 11) & 0x1F) * 255 / 31);
	rgb[1] = (uint8_t)(((c >> 5) & 0x3F) * 255 / 63);
	rgb[2] = (uint8_t)((c & 0x1F) * 255 / 31);
}

static void bc1_decode_block(const uint8_t *block, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint8_t *out_rgba) {
	uint16_t c0 = (uint16_t)(block[0] | (block[1] << 8));
	uint16_t c1 = (uint16_t)(block[2] | (block[3] << 8));
	uint32_t indices = (uint32_t)(block[4] | (block[5] << 8) | (block[6] << 16) | (block[7] << 24));

	uint8_t colors[4][3];
	bc1_unpack_565(c0, colors[0]);
	bc1_unpack_565(c1, colors[1]);
	if (c0 > c1) {
		for (int i = 0; i < 3; i++) {
			colors[2][i] = (uint8_t)((2 * colors[0][i] + colors[1][i]) / 3);
			colors[3][i] = (uint8_t)((colors[0][i] + 2 * colors[1][i]) / 3);
		}
	} else {
		for (int i = 0; i < 3; i++) {
			colors[2][i] = (uint8_t)((colors[0][i] + colors[1][i]) / 2);
			colors[3][i] = 0;
		}
	}

	for (uint32_t py = 0; py < 4; py++) {
		uint32_t y = y0 + py;
		if (y >= height) continue;
		for (uint32_t px = 0; px < 4; px++) {
			uint32_t x = x0 + px;
			if (x >= width) continue;
			uint32_t idx = (indices >> (2 * (py * 4 + px))) & 0x3u;
			uint8_t *dst = out_rgba + (size_t) (y * width + x) * 4;
			dst[0] = colors[idx][0];
			dst[1] = colors[idx][1];
			dst[2] = colors[idx][2];
			dst[3] = 255;
		}
	}
}

static void bc1_decode(const uint8_t *src, uint32_t width, uint32_t height, uint32_t src_row_pitch, uint8_t *out_rgba) {
	for (uint32_t by = 0; by * 4 < height; by++) {
		for (uint32_t bx = 0; bx * 4 < width; bx++) {
			const uint8_t *block = src + (size_t) by * src_row_pitch + (size_t) bx * 8;
			bc1_decode_block(block, bx * 4, by * 4, width, height, out_rgba);
		}
	}
}

// ---------------------------------------------------------------------------
// HSL -> RGB, for the procedurally-generated material textures (same
// rainbow-gradient technique as the original sample).
// ---------------------------------------------------------------------------

static float hsl_hue_to_rgb(float p, float q, float t) {
	if (t < 0.0f) t += 1.0f;
	if (t > 1.0f) t -= 1.0f;
	if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
	if (t < 1.0f / 2.0f) return q;
	if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
	return p;
}

static void hsl_to_rgb8(float h, float s, float l, uint8_t out[3]) {
	float r, g, b;
	if (s == 0.0f) {
		r = g = b = l;
	} else {
		float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
		float p = 2.0f * l - q;
		r = hsl_hue_to_rgb(p, q, h + 1.0f / 3.0f);
		g = hsl_hue_to_rgb(p, q, h);
		b = hsl_hue_to_rgb(p, q, h - 1.0f / 3.0f);
	}
	out[0] = (uint8_t)(r * 255.0f);
	out[1] = (uint8_t)(g * 255.0f);
	out[2] = (uint8_t)(b * 255.0f);
}

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


static void *read_entire_file(const char *path, size_t *out_size) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open file: %s\n", path);
		exit(1);
	}
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	void *buffer = malloc((size_t) size);
	fread(buffer, 1, (size_t) size, f);
	fclose(f);
	*out_size = (size_t) size;
	return buffer;
}

static spudgpu_shader_module load_shader_module(
    spudgpu_device device,
    const char *path,
    SPUDGPU_SHADER_STAGE stage) {
	size_t code_size;
	void *code = read_entire_file(path, &code_size);

	spudgpu_shader_module_desc desc = {
	    .stage      = stage,
	    .spirv_code = code,
	    .spirv_size = code_size,
	};
	spudgpu_shader_module module = NULL;
	SPUDRESULT result            = spudgpu_create_shader_module(device, &desc, &module);
	free(code);
	if (SPUDFAIL(result)) {
		fprintf(stderr, "spudgpu_create_shader_module failed for %s: %d\n", path, result);
		exit(1);
	}
	return module;
}

// Uploads `pixels` (tightly packed, width*height*4 bytes) into a freshly
// created SPUDGPU_FORMAT_R8G8B8A8_UNORM image, via a staging buffer sized
// and strided exactly as spudgpu_get_image_buffer_copy_size reports --
// portable across Vulkan (no alignment requirement) and D3D12 (256-byte row
// pitch alignment), rather than assuming a tightly-packed row pitch.
static spudgpu_image upload_rgba8_texture(
    spudgpu_device device,
    spudgpu_command_list cmd,
    const uint8_t *pixels,
    uint32_t width,
    uint32_t height,
    spudgpu_buffer *out_staging_buffer) {
	spudgpu_image_desc image_desc = {
	    .usage        = SPUDGPU_IMAGE_USAGE_SAMPLED | SPUDGPU_IMAGE_USAGE_TRANSFER_DST,
	    .type         = SPUDGPU_IMAGE_TYPE_2D,
	    .format       = SPUDGPU_FORMAT_R8G8B8A8_UNORM,
	    .width        = width,
	    .height       = height,
	    .depth        = 1,
	    .array_layers = 1,
	    .mip_levels   = 1,
	};
	spudgpu_image image = NULL;
	if (SPUDFAIL(spudgpu_create_image(device, &image_desc, &image))) {
		fprintf(stderr, "spudgpu_create_image failed\n");
		exit(1);
	}

	uint64_t row_pitch = 0, total_size = 0;
	spudgpu_get_image_buffer_copy_size(image, 0, &row_pitch, &total_size);
	if (row_pitch == 0) row_pitch = (uint64_t) width * 4;
	if (total_size == 0) total_size = row_pitch * height;

	spudgpu_buffer_desc staging_desc = {
	    .usage        = SPUDGPU_BUFFER_USAGE_TRANSFER_SRC,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = total_size,
	};
	spudgpu_buffer staging = NULL;
	if (SPUDFAIL(spudgpu_create_buffer(device, &staging_desc, &staging))) {
		fprintf(stderr, "spudgpu_create_buffer (staging) failed\n");
		exit(1);
	}

	uint8_t *mapped = NULL;
	spudgpu_map_buffer(staging, 0, 0, (void **) &mapped);
	for (uint32_t y = 0; y < height; y++)
		memcpy(mapped + y * row_pitch, pixels + (size_t) y * width * 4, (size_t) width * 4);
	spudgpu_flush_buffer(staging, 0, 0);
	spudgpu_unmap_buffer(staging);

	spudgpu_cmd_image_barrier(cmd, image, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_TRANSFER_DST);

	spudgpu_image_buffer_copy_desc copy_desc = {
	    .buffer_row_length   = (uint32_t) (row_pitch / 4),
	    .buffer_image_height = height,
	    .array_layer_count   = 1,
	    .width               = width,
	    .height              = height,
	    .depth               = 1,
	};
	spudgpu_cmd_copy_buffer_to_image(cmd, staging, image, &copy_desc);

	spudgpu_cmd_image_barrier(cmd, image, SPUDGPU_IMAGE_LAYOUT_TRANSFER_DST, SPUDGPU_IMAGE_LAYOUT_SHADER_READ_ONLY);

	// Must stay alive until the GPU has finished the copy above -- the
	// caller destroys *out_staging_buffer only after waiting for this
	// upload command list to complete, same "keep alive past Close(), free
	// after WaitForGpu" pattern the original sample uses for its upload
	// heaps.
	*out_staging_buffer = staging;
	return image;
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

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	SDL_Window *window = SDL_CreateWindow("SpudGPU DynamicIndexing", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		return 1;
	}

	spudgpu_instance instance = NULL;
	if (SPUDFAIL(spudgpu_create_instance("SpudGPUDynamicIndexing", 1, "SpudGPUSamples", 1, &instance))) {
		fprintf(stderr, "spudgpu_create_instance failed\n");
		return 1;
	}

	spudgpu_device *devices = NULL;
	uint32_t device_count   = 0;
	if (SPUDFAIL(spudgpu_enumerate_devices(instance, &devices, &device_count)) || device_count == 0) {
		fprintf(stderr, "spudgpu_enumerate_devices failed or returned no devices\n");
		return 1;
	}
	if (device_index >= device_count) {
		fprintf(stderr, "--device %u: there are %u devices\n", device_index, device_count);
		return 1;
	}
	spudgpu_device device = devices[device_index];

	SPUDGPU_DEVICE_PROPERTIES device_properties = {0};
	spudgpu_get_device_properties(device, &device_properties);
	printf("Using device: %s\n", device_properties.description);

	spudgpu_bindless_capabilities bindless_caps = {0};
	spudgpu_get_bindless_capabilities(device, &bindless_caps);
	if (!bindless_caps.supported) {
		fprintf(stderr, "This device/driver doesn't support bindless descriptor indexing\n");
		return 1;
	}

	spudgpu_command_queue graphics_queue = spudgpu_get_graphics_queue(device);

	spudgpu_surface surface = spudgpu_create_surface_from_sdl3(instance, window);
	if (!surface) {
		fprintf(stderr, "spudgpu_create_surface_from_sdl3 failed\n");
		return 1;
	}

	spudgpu_swap_chain_desc swap_chain_desc = {
	    .surface         = surface,
	    .queue           = graphics_queue,
	    .width           = WINDOW_WIDTH,
	    .height          = WINDOW_HEIGHT,
	    .buffer_count    = 2,
	    .format          = SPUDGPU_FORMAT_B8G8R8A8_UNORM,
	    .present_mode    = SPUDGPU_PRESENT_MODE_FIFO,
	    .fullscreen_mode = SPUDGPU_FULLSCREEN_MODE_WINDOWED,
	};
	spudgpu_swap_chain swap_chain = NULL;
	if (SPUDFAIL(spudgpu_create_swap_chain(device, &swap_chain_desc, &swap_chain))) {
		fprintf(stderr, "spudgpu_create_swap_chain failed\n");
		return 1;
	}

	spudgpu_command_allocator_desc allocator_desc = {.type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_command_allocator command_allocator   = NULL;
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_command_list cmd = NULL;
	spudgpu_create_command_list(command_allocator, &cmd);

	// ------------------------------------------------------------------
	// Mesh (occcity.bin).
	// ------------------------------------------------------------------

	size_t mesh_data_size;
	uint8_t *mesh_data = (uint8_t *) read_entire_file("occcity.bin", &mesh_data_size);

	spudgpu_buffer_desc vertex_buffer_desc = {
	    .usage        = SPUDGPU_BUFFER_USAGE_VERTEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = VERTEX_DATA_SIZE,
	};
	spudgpu_buffer vertex_buffer = NULL;
	spudgpu_create_buffer(device, &vertex_buffer_desc, &vertex_buffer);
	void *vertex_mapped = NULL;
	spudgpu_map_buffer(vertex_buffer, 0, 0, &vertex_mapped);
	memcpy(vertex_mapped, mesh_data + VERTEX_DATA_OFFSET, VERTEX_DATA_SIZE);
	spudgpu_flush_buffer(vertex_buffer, 0, 0);
	spudgpu_unmap_buffer(vertex_buffer);

	spudgpu_buffer_view_desc vertex_view_desc = {
	    .parent_buffer = vertex_buffer, .offset_from_parent_buffer = 0, .stride = VERTEX_STRIDE, .size = VERTEX_DATA_SIZE};
	spudgpu_buffer_view vertex_buffer_view = NULL;
	spudgpu_create_buffer_view(vertex_buffer, &vertex_view_desc, &vertex_buffer_view);

	spudgpu_buffer_desc index_buffer_desc = {
	    .usage        = SPUDGPU_BUFFER_USAGE_INDEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = INDEX_DATA_SIZE,
	};
	spudgpu_buffer index_buffer = NULL;
	spudgpu_create_buffer(device, &index_buffer_desc, &index_buffer);
	void *index_mapped = NULL;
	spudgpu_map_buffer(index_buffer, 0, 0, &index_mapped);
	memcpy(index_mapped, mesh_data + INDEX_DATA_OFFSET, INDEX_DATA_SIZE);
	spudgpu_flush_buffer(index_buffer, 0, 0);
	spudgpu_unmap_buffer(index_buffer);

	spudgpu_buffer_view_desc index_view_desc = {
	    .parent_buffer = index_buffer, .offset_from_parent_buffer = 0, .stride = 4, .size = INDEX_DATA_SIZE};
	spudgpu_buffer_view index_buffer_view = NULL;
	spudgpu_create_buffer_view(index_buffer, &index_view_desc, &index_buffer_view);

	uint32_t index_count = INDEX_DATA_SIZE / 4;

	// ------------------------------------------------------------------
	// Textures + sampler. Uploads happen on `cmd`, submitted once and
	// waited on below before the render loop starts.
	// ------------------------------------------------------------------

	spudgpu_begin_command_list(cmd);

	// One staging buffer per uploaded texture (1 diffuse + CITY_MATERIAL_COUNT
	// materials), kept alive until the upload command list below is waited on.
	spudgpu_buffer staging_buffers[1 + CITY_MATERIAL_COUNT];
	uint32_t staging_buffer_count = 0;

	uint8_t *diffuse_rgba = (uint8_t *) malloc((size_t) DIFFUSE_TEXTURE_WIDTH * DIFFUSE_TEXTURE_HEIGHT * 4);
	bc1_decode(mesh_data + DIFFUSE_BC1_OFFSET, DIFFUSE_TEXTURE_WIDTH, DIFFUSE_TEXTURE_HEIGHT, DIFFUSE_BC1_ROW_PITCH, diffuse_rgba);
	spudgpu_image diffuse_image = upload_rgba8_texture(
	    device, cmd, diffuse_rgba, DIFFUSE_TEXTURE_WIDTH, DIFFUSE_TEXTURE_HEIGHT, &staging_buffers[staging_buffer_count++]);
	free(diffuse_rgba);
	free(mesh_data);

	spudgpu_image_view_desc diffuse_view_desc = {
	    .parent_image = diffuse_image,
	    .type         = SPUDGPU_IMAGE_VIEW_TYPE_2D,
	    .subresource_range = {.aspect_mask = 1 /* COLOR */, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
	};
	spudgpu_image_view diffuse_view = NULL;
	spudgpu_create_image_view(diffuse_image, &diffuse_view_desc, &diffuse_view);

	uint32_t diffuse_bindless_index = SPUDGPU_BINDLESS_INVALID_INDEX;
	spudgpu_bindless_register_sampled_image(device, diffuse_view, &diffuse_bindless_index);

	spudgpu_image material_images[CITY_MATERIAL_COUNT];
	spudgpu_image_view material_views[CITY_MATERIAL_COUNT];
	uint32_t material_bindless_index[CITY_MATERIAL_COUNT];

	uint8_t *material_rgba = (uint8_t *) malloc((size_t) CITY_MATERIAL_TEXTURE_WIDTH * CITY_MATERIAL_TEXTURE_HEIGHT * 4);
	float grad_step = 1.0f / (float) CITY_MATERIAL_COUNT;
	for (uint32_t i = 0; i < CITY_MATERIAL_COUNT; i++) {
		float t = (float) i * grad_step;
		for (uint32_t y = 0; y < CITY_MATERIAL_TEXTURE_HEIGHT; y++) {
			float t_row = t + ((float) y / (float) CITY_MATERIAL_TEXTURE_HEIGHT) * grad_step;
			uint8_t rgb[3];
			hsl_to_rgb8(t_row, 0.5f, 0.5f, rgb);
			for (uint32_t x = 0; x < CITY_MATERIAL_TEXTURE_WIDTH; x++) {
				uint8_t *dst = material_rgba + (size_t) (y * CITY_MATERIAL_TEXTURE_WIDTH + x) * 4;
				dst[0] = rgb[0];
				dst[1] = rgb[1];
				dst[2] = rgb[2];
				dst[3] = 255;
			}
		}

		material_images[i] = upload_rgba8_texture(
		    device, cmd, material_rgba, CITY_MATERIAL_TEXTURE_WIDTH, CITY_MATERIAL_TEXTURE_HEIGHT, &staging_buffers[staging_buffer_count++]);

		spudgpu_image_view_desc mat_view_desc = {
		    .parent_image = material_images[i],
		    .type         = SPUDGPU_IMAGE_VIEW_TYPE_2D,
		    .subresource_range = {.aspect_mask = 1, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
		};
		spudgpu_create_image_view(material_images[i], &mat_view_desc, &material_views[i]);
		spudgpu_bindless_register_sampled_image(device, material_views[i], &material_bindless_index[i]);
	}
	free(material_rgba);

	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);
	for (uint32_t i = 0; i < staging_buffer_count; i++)
		spudgpu_destroy_buffer(staging_buffers[i]);

	spudgpu_sampler_desc sampler_desc = {
	    .mag_filter      = SPUDGPU_FILTER_LINEAR,
	    .min_filter      = SPUDGPU_FILTER_LINEAR,
	    .mipmap_filter   = SPUDGPU_FILTER_LINEAR,
	    .address_mode_u  = SPUDGPU_ADDRESS_MODE_REPEAT,
	    .address_mode_v  = SPUDGPU_ADDRESS_MODE_REPEAT,
	    .address_mode_w  = SPUDGPU_ADDRESS_MODE_REPEAT,
	    .max_lod         = 1000.0f,
	    .max_anisotropy  = 1.0f,
	};
	spudgpu_sampler sampler = NULL;
	if (SPUDFAIL(spudgpu_create_sampler(device, &sampler_desc, &sampler))) {
		fprintf(stderr, "spudgpu_create_sampler failed\n");
		return 1;
	}

	// ------------------------------------------------------------------
	// Descriptor sets: 0 = bindless (textures), 1 = sampler, 2 = per-building CBV.
	// ------------------------------------------------------------------

	spudgpu_descriptor_set_layout bindless_layout = spudgpu_get_bindless_descriptor_set_layout(device);

	spudgpu_descriptor_set_layout_desc sampler_layout_desc = {
	    .bindings      = {{.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT}},
	    .binding_count = 1,
	};
	spudgpu_descriptor_set_layout sampler_layout = NULL;
	spudgpu_create_descriptor_set_layout(device, &sampler_layout_desc, &sampler_layout);

	spudgpu_descriptor_pool_desc sampler_pool_desc = {
	    .max_sets = 1, .pool_sizes = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = 1}}, .pool_size_count = 1};
	spudgpu_descriptor_pool sampler_pool = NULL;
	spudgpu_create_descriptor_pool(device, &sampler_pool_desc, &sampler_pool);

	spudgpu_descriptor_set_desc sampler_set_desc = {.pool = sampler_pool, .set_layouts = {sampler_layout}, .set_count = 1};
	spudgpu_descriptor_set sampler_set = NULL;
	spudgpu_create_descriptor_sets(device, &sampler_set_desc, &sampler_set);

	spudgpu_write_descriptor_set sampler_write = {
	    .dst_set = sampler_set, .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .sampler = sampler};
	spudgpu_update_descriptor_sets(device, &sampler_write, 1);

	spudgpu_descriptor_set_layout_desc cbv_layout_desc = {
	    .bindings      = {{.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_VERTEX}},
	    .binding_count = 1,
	};
	spudgpu_descriptor_set_layout cbv_layout = NULL;
	spudgpu_create_descriptor_set_layout(device, &cbv_layout_desc, &cbv_layout);

	spudgpu_descriptor_pool_desc cbv_pool_desc = {
	    .max_sets = CITY_MATERIAL_COUNT, .pool_sizes = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = CITY_MATERIAL_COUNT}}, .pool_size_count = 1};
	spudgpu_descriptor_pool cbv_pool = NULL;
	spudgpu_create_descriptor_pool(device, &cbv_pool_desc, &cbv_pool);

	const uint32_t cbv_stride = 256; // D3D12 CBV alignment; harmless elsewhere.
	spudgpu_buffer_desc cbv_buffer_desc = {
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = (uint64_t) cbv_stride * CITY_MATERIAL_COUNT,
	};
	spudgpu_buffer cbv_buffer = NULL;
	spudgpu_create_buffer(device, &cbv_buffer_desc, &cbv_buffer);
	void *cbv_mapped = NULL;
	spudgpu_map_buffer(cbv_buffer, 0, 0, &cbv_mapped);

	spudgpu_descriptor_set cbv_sets[CITY_MATERIAL_COUNT];
	float model_matrices[CITY_MATERIAL_COUNT][16];
	for (uint32_t row = 0; row < CITY_ROW_COUNT; row++) {
		for (uint32_t col = 0; col < CITY_COLUMN_COUNT; col++) {
			uint32_t i = row * CITY_COLUMN_COUNT + col;

			mat4_translation(col * CITY_SPACING_INTERVAL, 0.02f * (float) i, -(float) row * CITY_SPACING_INTERVAL, model_matrices[i]);

			spudgpu_descriptor_set_desc set_desc = {.pool = cbv_pool, .set_layouts = {cbv_layout}, .set_count = 1};
			spudgpu_create_descriptor_sets(device, &set_desc, &cbv_sets[i]);

			spudgpu_descriptor_buffer_info buf_info = {.buffer = cbv_buffer, .offset = (uint64_t) i * cbv_stride, .range = sizeof(float) * 16};
			spudgpu_write_descriptor_set write = {
			    .dst_set = cbv_sets[i], .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer_info = &buf_info};
			spudgpu_update_descriptor_sets(device, &write, 1);
		}
	}

	// ------------------------------------------------------------------
	// Pipeline + depth buffer.
	// ------------------------------------------------------------------

	spudgpu_shader_module vertex_module   = load_shader_module(device, "shaders/scene.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	spudgpu_shader_module fragment_module = load_shader_module(device, "shaders/scene.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);

	spudgpu_shader_pipeline_desc pipeline_desc = {
	    .vertex_module   = vertex_module,
	    .fragment_module = fragment_module,
	    .vertex_attributes = {
	        {.location = 0, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, position)},
	        {.location = 1, .binding = 0, .format = SPUDGPU_FORMAT_R32G32_FLOAT, .offset = offsetof(Vertex, uv)},
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
	    .descriptor_set_layouts      = {bindless_layout, sampler_layout, cbv_layout},
	    .descriptor_set_layout_count = 3,
	    .push_constant_ranges        = {{.offset = 0, .size = sizeof(PushConstants), .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT}},
	    .push_constant_range_count   = 1,
	};
	spudgpu_shader_pipeline pipeline = NULL;
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		return 1;
	}

	spudgpu_image_desc depth_image_desc = {
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
	spudgpu_image depth_image = NULL;
	if (SPUDFAIL(spudgpu_create_image(device, &depth_image_desc, &depth_image))) {
		fprintf(stderr, "spudgpu_create_image (depth) failed\n");
		return 1;
	}
	spudgpu_image_view_desc depth_view_desc = {
	    .parent_image = depth_image,
	    .type         = SPUDGPU_IMAGE_VIEW_TYPE_2D,
	    .subresource_range = {.aspect_mask = SPUDGPU_ASPECT_DEPTH_BIT, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
	};
	spudgpu_image_view depth_view = NULL;
	spudgpu_create_image_view(depth_image, &depth_view_desc, &depth_view);

	// ------------------------------------------------------------------
	// Main loop.
	// ------------------------------------------------------------------

	Camera camera;
	camera_init(&camera, (CITY_COLUMN_COUNT / 2.0f) * CITY_SPACING_INTERVAL - (CITY_SPACING_INTERVAL / 2.0f), 15.0f, 50.0f, CITY_SPACING_INTERVAL * 2.0f);

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

		float view[16], proj[16], view_proj[16];
		mat4_look_to_rh(camera.position, camera.look_dir, up, view);
		mat4_perspective_rh(0.8f, (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT, 1.0f, 1000.0f, proj);
		mat4_multiply(proj, view, view_proj);

		for (uint32_t i = 0; i < CITY_MATERIAL_COUNT; i++) {
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
		// Depth is cleared every pass, so it comes from UNDEFINED each frame,
		// like the backbuffer. It was once transitioned at setup, into the
		// upload list after that list had been closed and submitted.
		spudgpu_cmd_image_barrier(cmd, depth_image, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

		spudgpu_rendering_begin_desc rendering_desc = {
		    .color_attachments = {{.image_view = backbuffer_view, .load_op = SPUDGPU_LOAD_OP_CLEAR, .store_op = SPUDGPU_STORE_OP_STORE, .clear_color = {0.0f, 0.2f, 0.4f, 1.0f}}},
		    .color_attachment_count = 1,
		    .depth_attachment       = {.image_view = depth_view, .depth_load_op = SPUDGPU_LOAD_OP_CLEAR, .depth_store_op = SPUDGPU_STORE_OP_STORE, .clear_depth = 1.0f},
		    .width                  = WINDOW_WIDTH,
		    .height                 = WINDOW_HEIGHT,
		};
		spudgpu_cmd_begin_rendering(cmd, &rendering_desc);

		spudgpu_cmd_bind_pipeline(cmd, pipeline);
		spudgpu_cmd_bind_bindless_resources(cmd, pipeline, 0);
		spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 1, &sampler_set, 1);

		SPUDGPU_VIEWPORT viewport = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT, .minDepth = 0.0f, .maxDepth = 1.0f};
		spudgpu_cmd_set_viewports(cmd, 0, 1, &viewport);
		SPUDGPU_SCISSOR_RECT scissor = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};
		spudgpu_cmd_set_scissor_rects(cmd, 0, 1, &scissor);

		spudgpu_cmd_set_vertex_buffers(cmd, 0, 1, &vertex_buffer_view);
		spudgpu_cmd_set_index_buffer(cmd, index_buffer_view);

		for (uint32_t i = 0; i < CITY_MATERIAL_COUNT; i++) {
			PushConstants pc = {.diffuse_index = diffuse_bindless_index, .material_index = material_bindless_index[i]};
			spudgpu_cmd_push_constants(cmd, pipeline, 0, sizeof(pc), &pc);
			spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 2, &cbv_sets[i], 1);
			spudgpu_cmd_draw_indexed(cmd, index_count, 0, 0);
		}

		spudgpu_cmd_end_rendering(cmd);
		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, SPUDGPU_IMAGE_LAYOUT_PRESENT_SRC);

		spudgpu_end_command_list(cmd);
		spudgpu_submit_command_lists_synced(graphics_queue, &cmd, 1, swap_chain);
		spudgpu_swap_chain_present(swap_chain);
	}

	spudgpu_queue_wait_idle(graphics_queue);

	spudgpu_destroy_image_view(depth_view);
	spudgpu_destroy_image(depth_image);
	spudgpu_destroy_shader_pipeline(pipeline);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(vertex_module);
	spudgpu_unmap_buffer(cbv_buffer);
	spudgpu_destroy_buffer(cbv_buffer);
	spudgpu_destroy_descriptor_pool(cbv_pool);
	spudgpu_destroy_descriptor_set_layout(cbv_layout);
	spudgpu_destroy_descriptor_pool(sampler_pool);
	spudgpu_destroy_descriptor_set_layout(sampler_layout);
	spudgpu_destroy_sampler(sampler);
	for (uint32_t i = 0; i < CITY_MATERIAL_COUNT; i++) {
		spudgpu_bindless_unregister_sampled_image(device, material_bindless_index[i]);
		spudgpu_destroy_image_view(material_views[i]);
		spudgpu_destroy_image(material_images[i]);
	}
	spudgpu_bindless_unregister_sampled_image(device, diffuse_bindless_index);
	spudgpu_destroy_image_view(diffuse_view);
	spudgpu_destroy_image(diffuse_image);
	spudgpu_destroy_buffer_view(index_buffer_view);
	spudgpu_destroy_buffer(index_buffer);
	spudgpu_destroy_buffer_view(vertex_buffer_view);
	spudgpu_destroy_buffer(vertex_buffer);
	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_swap_chain(swap_chain);
	spudgpu_destroy_surface(surface);
	spudgpu_destroy_instance(instance);

	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
