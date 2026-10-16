//
// SpudGPU port of D3D12SmallResources from d3d12samples, named for the
// SpudGPU feature it shows: heaps and the resources placed in them
// (spudgpu_create_heap, spudgpu_get_*_placement_requirements,
// spudgpu_create_placed_buffer/image) - see ../../README.md.
//
// An 11 x 7 grid of small (32 x 32) checkerboard textures, each drawn on its
// own quad. Space switches between the two ways of making the textures:
// placed, one after another in a single heap, or committed, one device
// allocation each - the switch D3D12SmallResources makes with
// m_usePlacedResources. The window title says which, and how much heap the
// placed textures take.
//
// Deviations from the original:
// - The vertex buffer is placed too, in a heap of device memory, and the
//   staging buffer that fills it and the textures is placed in a heap of
//   upload memory where the device has one (SPUDGPU_DEVICE_PROPERTIES::
//   mappable_heaps; not a Mac without Apple silicon, where it falls back to
//   spudgpu_create_buffer). The original has no buffer heaps. This way the
//   sample places every kind of resource SpudGPU can place.
// - The title shows the heap's size, not the adapter's memory use: SpudGPU
//   has no memory budget query (QueryVideoMemoryInfo,
//   VK_EXT_memory_budget, MTLDevice.currentAllocatedSize) yet. So the
//   committed mode can't say what it costs.
// - Textures are placed at the alignment the heap reports. On D3D12 that is
//   64 KiB per texture today, as committed: SpudGPU doesn't ask for
//   D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT (4 KiB) yet, which is the
//   original's whole point (see spudlib/CLAUDE.md). On Vulkan and Metal the
//   placed textures pack as tightly as the driver allows.
// - One upload on the graphics queue instead of a copy queue, as in
//   ../HelloTexture, and one staging buffer for all textures, filled once,
//   instead of one per texture per switch.
// - A dynamic sampler and separate SAMPLED_IMAGE + SAMPLER descriptors, for
//   the same reasons as ../HelloTexture (see its header). One descriptor set
//   per texture stands in for the original's SRV table slot per texture.
//

#include <SDL3/SDL.h>
#include <spudgpu.h>
#include <spudgpu_sdl3.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

#define GRID_WIDTH 11
#define GRID_HEIGHT 7
#define TEXTURE_COUNT (GRID_WIDTH * GRID_HEIGHT)
#define TEXTURE_WIDTH 32
#define TEXTURE_HEIGHT 32
#define TEXTURE_PIXEL_SIZE 4

// Where texture data starts in a buffer: D3D12's
// D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, the strictest of the three APIs.
#define TEXTURE_DATA_ALIGNMENT 512u

typedef struct Vertex {
	float position[3];
	float uv[2];
} Vertex;

static void check(SPUDRESULT result, const char *what) {
	if (SPUDFAIL(result)) {
		fprintf(stderr, "%s failed: %d\n", what, result);
		exit(1);
	}
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
	return (value + alignment - 1) / alignment * alignment;
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
	check(result, path);
	return module;
}

// D3D12SmallResources::GenerateTexture: an 8 x 8 checkerboard of black and
// one random color. The caller seeds rand() so the colors come out the same
// every time.
static void generate_texture(uint8_t *pixels) {
	const uint32_t row_pitch    = TEXTURE_WIDTH * TEXTURE_PIXEL_SIZE;
	const uint32_t cell_pitch   = row_pitch >> 3;
	const uint32_t cell_height  = TEXTURE_WIDTH >> 3;
	const uint32_t texture_size = row_pitch * TEXTURE_HEIGHT;

	const uint8_t r = (uint8_t) (rand() & 0xff);
	const uint8_t g = (uint8_t) (rand() & 0xff);
	const uint8_t b = (uint8_t) (rand() & 0xff);
	for (uint32_t n = 0; n < texture_size; n += TEXTURE_PIXEL_SIZE) {
		const uint32_t i = (n % row_pitch) / cell_pitch;
		const uint32_t j = (n / row_pitch) / cell_height;
		const bool black = i % 2 == j % 2;
		pixels[n]        = black ? 0x00 : r;
		pixels[n + 1]    = black ? 0x00 : g;
		pixels[n + 2]    = black ? 0x00 : b;
		pixels[n + 3]    = 0xff;
	}
}

// Rounds a heap size up to what spudgpu_create_heap takes.
static uint64_t heap_size_for(uint64_t bytes) {
	return align_up(bytes, SPUDGPU_HEAP_SIZE_GRANULARITY);
}

static const spudgpu_image_desc texture_desc = {
    .usage        = SPUDGPU_IMAGE_USAGE_SAMPLED | SPUDGPU_IMAGE_USAGE_TRANSFER_DST,
    .type         = SPUDGPU_IMAGE_TYPE_2D,
    .format       = SPUDGPU_FORMAT_R8G8B8A8_UNORM,
    .width        = TEXTURE_WIDTH,
    .height       = TEXTURE_HEIGHT,
    .depth        = 1,
    .array_layers = 1,
    .mip_levels   = 1,
};

// The textures of one mode, and the heap they are in if they are placed.
typedef struct Textures {
	bool placed;
	spudgpu_heap heap; // NULL when committed
	spudgpu_placement_requirements requirements;
	uint64_t stride; // Bytes from one placed texture to the next
	spudgpu_image images[TEXTURE_COUNT];
	spudgpu_image_view views[TEXTURE_COUNT];
} Textures;

// D3D12SmallResources::CreateTextures, without the upload: the images, in
// one heap when [placed] and one allocation each when not.
static void create_textures(spudgpu_device device, bool placed, Textures *out) {
	memset(out, 0, sizeof(*out));
	out->placed = placed;

	if (placed) {
		// A heap is needed to ask what a texture needs from one, and its
		// size depends on the answer: ask a small heap of the same kind and
		// class first. The answer is the same for every heap of that kind
		// and class on a device.
		spudgpu_heap_desc probe_desc = {
		    .memory_kind    = SPUDGPU_MEMORY_KIND_DEVICE,
		    .resource_class = SPUDGPU_HEAP_RESOURCE_CLASS_IMAGES,
		    .size           = SPUDGPU_HEAP_SIZE_GRANULARITY,
		};
		spudgpu_heap probe = NULL;
		check(spudgpu_create_heap(device, &probe_desc, &probe), "spudgpu_create_heap (probe)");
		check(spudgpu_get_image_placement_requirements(probe, &texture_desc, &out->requirements),
		      "spudgpu_get_image_placement_requirements");
		spudgpu_destroy_heap(probe);

		// The textures one after another, each at the next offset its
		// alignment allows.
		out->stride = align_up(out->requirements.size, out->requirements.alignment);
		spudgpu_heap_desc heap_desc = {
		    .memory_kind    = SPUDGPU_MEMORY_KIND_DEVICE,
		    .resource_class = SPUDGPU_HEAP_RESOURCE_CLASS_IMAGES,
		    .size           = heap_size_for(out->stride * TEXTURE_COUNT),
		};
		check(spudgpu_create_heap(device, &heap_desc, &out->heap), "spudgpu_create_heap (textures)");

		for (uint32_t n = 0; n < TEXTURE_COUNT; n++)
			check(spudgpu_create_placed_image(out->heap, n * out->stride, &texture_desc, &out->images[n]),
			      "spudgpu_create_placed_image");
	} else {
		for (uint32_t n = 0; n < TEXTURE_COUNT; n++)
			check(spudgpu_create_image(device, &texture_desc, &out->images[n]), "spudgpu_create_image");
	}

	for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
		spudgpu_image_view_desc view_desc = {
		    .parent_image      = out->images[n],
		    .type              = SPUDGPU_IMAGE_VIEW_TYPE_2D,
		    .subresource_range = {.aspect_mask = SPUDGPU_IMAGE_ASPECT_COLOR, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
		};
		check(spudgpu_create_image_view(out->images[n], &view_desc, &out->views[n]), "spudgpu_create_image_view");
	}
}

// Every placed image is destroyed before its heap.
static void destroy_textures(Textures *textures) {
	for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
		spudgpu_destroy_image_view(textures->views[n]);
		spudgpu_destroy_image(textures->images[n]);
	}
	spudgpu_destroy_heap(textures->heap);
	memset(textures, 0, sizeof(*textures));
}

// Records the copy of every texture's pixels from [staging], where texture
// n starts at texture_offsets[n] with rows [row_pitch] apart.
static void record_texture_upload(
    spudgpu_command_list cmd,
    const Textures *textures,
    spudgpu_buffer staging,
    const uint64_t *texture_offsets,
    uint64_t row_pitch) {
	for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
		// A placed image's contents are undefined, as a new image's are:
		// UNDEFINED is the honest old layout either way.
		spudgpu_cmd_image_barrier(cmd, textures->images[n], SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_TRANSFER_DST);
		spudgpu_image_buffer_copy_desc copy_desc = {
		    .buffer_offset       = texture_offsets[n],
		    .buffer_row_length   = (uint32_t) (row_pitch / TEXTURE_PIXEL_SIZE),
		    .buffer_image_height = TEXTURE_HEIGHT,
		    .array_layer_count   = 1,
		    .width               = TEXTURE_WIDTH,
		    .height              = TEXTURE_HEIGHT,
		    .depth               = 1,
		};
		spudgpu_cmd_copy_buffer_to_image(cmd, staging, textures->images[n], &copy_desc);
		spudgpu_cmd_image_barrier(cmd, textures->images[n], SPUDGPU_IMAGE_LAYOUT_TRANSFER_DST, SPUDGPU_IMAGE_LAYOUT_SHADER_READ_ONLY);
	}
}

// Points each texture's descriptor set at its view.
static void write_texture_sets(spudgpu_device device, const Textures *textures, spudgpu_descriptor_set *sets, spudgpu_sampler sampler) {
	for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
		spudgpu_descriptor_image_info image_info = {
		    .image_view   = textures->views[n],
		    .image_layout = SPUDGPU_IMAGE_LAYOUT_SHADER_READ_ONLY,
		};
		spudgpu_write_descriptor_set writes[2] = {
		    {.dst_set = sets[n], .dst_binding = 0, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .image_info = &image_info},
		    {.dst_set = sets[n], .dst_binding = 1, .descriptor_count = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .sampler = sampler},
		};
		spudgpu_update_descriptor_sets(device, writes, 2);
	}
}

static void set_title(SDL_Window *window, const Textures *textures) {
	char title[256];
	if (textures->placed) {
		snprintf(title, sizeof(title),
		         "SpudGPU Heaps [Placed] - %u textures in one %llu KB heap (%llu B each, %llu B alignment) - Space: committed",
		         TEXTURE_COUNT,
		         (unsigned long long) (heap_size_for(textures->stride * TEXTURE_COUNT) / 1024),
		         (unsigned long long) textures->requirements.size,
		         (unsigned long long) textures->requirements.alignment);
	} else {
		snprintf(title, sizeof(title),
		         "SpudGPU Heaps [Committed] - %u textures, one device allocation each - Space: placed",
		         TEXTURE_COUNT);
	}
	SDL_SetWindowTitle(window, title);
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

	SDL_Window *window = SDL_CreateWindow("SpudGPU Heaps", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		return 1;
	}

	spudgpu_instance instance = NULL;
	check(spudgpu_create_instance("SpudGPUHeaps", 1, "SpudGPUSamples", 1, &instance), "spudgpu_create_instance");

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
	printf("Using device: %s (mappable heaps: %s)\n", device_properties.description,
	       device_properties.mappable_heaps ? "yes" : "no");

	spudgpu_command_queue graphics_queue = spudgpu_get_graphics_queue(device);

	spudgpu_surface surface = spudgpu_create_surface_from_sdl3(instance, window);
	if (!surface) {
		fprintf(stderr, "spudgpu_create_surface_from_sdl3 failed\n");
		return 1;
	}

	spudgpu_swap_chain_desc swap_chain_desc = {
	    .surface = surface,
	    .queue   = graphics_queue,
	    .width   = WINDOW_WIDTH,
	    .height  = WINDOW_HEIGHT,
	    // Metal hands out one drawable at a time - see ../HelloTexture.
#if SPUDGPU_COMPILE_METAL_API
	    .buffer_count = 1,
#else
	    .buffer_count = 2,
#endif
	    .format          = SPUDGPU_FORMAT_B8G8R8A8_UNORM,
	    .present_mode    = SPUDGPU_PRESENT_MODE_FIFO,
	    .fullscreen_mode = SPUDGPU_FULLSCREEN_MODE_WINDOWED,
	};
	spudgpu_swap_chain swap_chain = NULL;
	check(spudgpu_create_swap_chain(device, &swap_chain_desc, &swap_chain), "spudgpu_create_swap_chain");

	spudgpu_command_allocator_desc allocator_desc = {.type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_command_allocator command_allocator   = NULL;
	check(spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator), "spudgpu_create_command_allocator");

	spudgpu_command_list cmd = NULL;
	check(spudgpu_create_command_list(command_allocator, &cmd), "spudgpu_create_command_list");

	// ------------------------------------------------------------------
	// The quads: four vertices each, drawn as a triangle strip, laid out
	// as in the original.
	// ------------------------------------------------------------------

	const float aspect_ratio = (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT;
	static Vertex quad_vertices[TEXTURE_COUNT * 4];
	{
		uint32_t index       = 0;
		const float offset_x = 0.15f;
		const float margin_x = offset_x / 10.0f;
		const float start_x  = (GRID_WIDTH / 2.0f) * -(offset_x + margin_x) + margin_x / 2.0f;
		const float offset_y = offset_x * aspect_ratio;
		const float margin_y = offset_y / 10.0f;
		float y              = (GRID_HEIGHT / 2.0f) * (offset_y + margin_y) - margin_y / 2.0f;
		for (uint32_t row = 0; row < GRID_HEIGHT; row++) {
			float x = start_x;
			for (uint32_t column = 0; column < GRID_WIDTH; column++) {
				quad_vertices[index++] = (Vertex) {{x, y - offset_y, 0.0f}, {0.0f, 0.0f}};
				quad_vertices[index++] = (Vertex) {{x, y, 0.0f}, {0.0f, 1.0f}};
				quad_vertices[index++] = (Vertex) {{x + offset_x, y - offset_y, 0.0f}, {1.0f, 0.0f}};
				quad_vertices[index++] = (Vertex) {{x + offset_x, y, 0.0f}, {1.0f, 1.0f}};
				x += offset_x + margin_x;
			}
			y -= offset_y + margin_y;
		}
	}

	// ------------------------------------------------------------------
	// The textures, placed to begin with as in the original. They are made
	// before the staging buffer because its layout depends on the row pitch
	// a copy into one of them needs.
	// ------------------------------------------------------------------

	Textures textures;
	create_textures(device, true, &textures);

	uint64_t row_pitch = 0, texture_copy_size = 0;
	spudgpu_get_image_buffer_copy_size(textures.images[0], 0, &row_pitch, &texture_copy_size);
	if (row_pitch == 0) row_pitch = (uint64_t) TEXTURE_WIDTH * TEXTURE_PIXEL_SIZE;
	if (texture_copy_size == 0) texture_copy_size = row_pitch * TEXTURE_HEIGHT;

	// Staging: the vertices first, then each texture at the next offset
	// texture data may start at.
	uint64_t texture_offsets[TEXTURE_COUNT];
	uint64_t staging_size = sizeof(quad_vertices);
	for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
		texture_offsets[n] = align_up(staging_size, TEXTURE_DATA_ALIGNMENT);
		staging_size       = texture_offsets[n] + texture_copy_size;
	}

	// ------------------------------------------------------------------
	// The staging buffer, placed in a heap of upload memory where the
	// device has one, and the vertex buffer, placed in a heap of device
	// memory. Each heap is asked what its buffer needs, and sized to fit.
	// ------------------------------------------------------------------

	const spudgpu_buffer_desc staging_desc = {
	    .usage       = SPUDGPU_BUFFER_USAGE_TRANSFER_SRC,
	    .memory_kind = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size        = staging_size,
	};
	spudgpu_heap staging_heap = NULL;
	spudgpu_buffer staging    = NULL;
	if (device_properties.mappable_heaps) {
		spudgpu_heap_desc probe_desc = {
		    .memory_kind    = SPUDGPU_MEMORY_KIND_UPLOAD,
		    .resource_class = SPUDGPU_HEAP_RESOURCE_CLASS_BUFFERS,
		    .size           = SPUDGPU_HEAP_SIZE_GRANULARITY,
		};
		spudgpu_heap probe = NULL;
		check(spudgpu_create_heap(device, &probe_desc, &probe), "spudgpu_create_heap (upload probe)");
		spudgpu_placement_requirements staging_requirements;
		check(spudgpu_get_buffer_placement_requirements(probe, &staging_desc, &staging_requirements),
		      "spudgpu_get_buffer_placement_requirements (staging)");
		spudgpu_destroy_heap(probe);

		spudgpu_heap_desc heap_desc = probe_desc;
		heap_desc.size              = heap_size_for(staging_requirements.size);
		check(spudgpu_create_heap(device, &heap_desc, &staging_heap), "spudgpu_create_heap (upload)");
		check(spudgpu_create_placed_buffer(staging_heap, 0, &staging_desc, &staging), "spudgpu_create_placed_buffer (staging)");
	} else {
		check(spudgpu_create_buffer(device, &staging_desc, &staging), "spudgpu_create_buffer (staging)");
	}

	{
		uint8_t *mapped = NULL;
		check(spudgpu_map_buffer(staging, 0, 0, (void **) &mapped), "spudgpu_map_buffer");
		memcpy(mapped, quad_vertices, sizeof(quad_vertices));

		// Colors are random; the seed is the original's, so they are the
		// same every run.
		srand(100);
		uint8_t pixels[TEXTURE_WIDTH * TEXTURE_HEIGHT * TEXTURE_PIXEL_SIZE];
		for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
			generate_texture(pixels);
			for (uint32_t y = 0; y < TEXTURE_HEIGHT; y++)
				memcpy(mapped + texture_offsets[n] + y * row_pitch,
				       pixels + (size_t) y * TEXTURE_WIDTH * TEXTURE_PIXEL_SIZE,
				       (size_t) TEXTURE_WIDTH * TEXTURE_PIXEL_SIZE);
		}
		check(spudgpu_flush_buffer(staging, 0, 0), "spudgpu_flush_buffer");
		spudgpu_unmap_buffer(staging);
	}

	const spudgpu_buffer_desc vertex_buffer_desc = {
	    .usage       = SPUDGPU_BUFFER_USAGE_VERTEX | SPUDGPU_BUFFER_USAGE_TRANSFER_DST,
	    .memory_kind = SPUDGPU_MEMORY_KIND_DEVICE,
	    .size        = sizeof(quad_vertices),
	};
	spudgpu_heap vertex_heap     = NULL;
	spudgpu_buffer vertex_buffer = NULL;
	{
		spudgpu_heap_desc heap_desc = {
		    .memory_kind    = SPUDGPU_MEMORY_KIND_DEVICE,
		    .resource_class = SPUDGPU_HEAP_RESOURCE_CLASS_BUFFERS,
		    .size           = SPUDGPU_HEAP_SIZE_GRANULARITY,
		};
		check(spudgpu_create_heap(device, &heap_desc, &vertex_heap), "spudgpu_create_heap (vertices)");
		spudgpu_placement_requirements vertex_requirements;
		check(spudgpu_get_buffer_placement_requirements(vertex_heap, &vertex_buffer_desc, &vertex_requirements),
		      "spudgpu_get_buffer_placement_requirements (vertices)");
		// 6 KB of vertices fits in the smallest heap there is; a caller
		// that didn't know that would size the heap from the answer.
		if (vertex_requirements.size > heap_desc.size) {
			fprintf(stderr, "vertex buffer needs %llu bytes, more than its heap\n",
			        (unsigned long long) vertex_requirements.size);
			return 1;
		}
		check(spudgpu_create_placed_buffer(vertex_heap, 0, &vertex_buffer_desc, &vertex_buffer), "spudgpu_create_placed_buffer (vertices)");
	}

	spudgpu_buffer_view_desc vertex_buffer_view_desc = {
	    .parent_buffer             = vertex_buffer,
	    .offset_from_parent_buffer = 0,
	    .stride                    = sizeof(Vertex),
	    .size                      = sizeof(quad_vertices),
	};
	spudgpu_buffer_view vertex_buffer_view = NULL;
	check(spudgpu_create_buffer_view(vertex_buffer, &vertex_buffer_view_desc, &vertex_buffer_view), "spudgpu_create_buffer_view");

	// ------------------------------------------------------------------
	// The first upload: vertices and textures.
	// ------------------------------------------------------------------

	spudgpu_begin_command_list(cmd);
	{
		spudgpu_buffer_barrier to_copy = {.buffer = vertex_buffer, .state_before = SPUDGPU_RESOURCE_STATE_COMMON, .state_after = SPUDGPU_RESOURCE_STATE_COPY_DEST};
		spudgpu_cmd_pipeline_barrier(cmd, &to_copy, 1, NULL, 0);
		spudgpu_cmd_copy_buffer(cmd, staging, vertex_buffer, 0, 0, sizeof(quad_vertices));
		spudgpu_buffer_barrier to_vertex = {.buffer = vertex_buffer, .state_before = SPUDGPU_RESOURCE_STATE_COPY_DEST, .state_after = SPUDGPU_RESOURCE_STATE_VERTEX_BUFFER};
		spudgpu_cmd_pipeline_barrier(cmd, &to_vertex, 1, NULL, 0);
	}
	record_texture_upload(cmd, &textures, staging, texture_offsets, row_pitch);
	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);

	// ------------------------------------------------------------------
	// Sampler, one descriptor set per texture, pipeline.
	// ------------------------------------------------------------------

	spudgpu_sampler_desc sampler_desc = {
	    .mag_filter     = SPUDGPU_FILTER_LINEAR,
	    .min_filter     = SPUDGPU_FILTER_LINEAR,
	    .mipmap_filter  = SPUDGPU_FILTER_LINEAR,
	    .address_mode_u = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .address_mode_v = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .address_mode_w = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .max_lod        = 0.0f,
	    .max_anisotropy = 1.0f,
	};
	spudgpu_sampler sampler = NULL;
	check(spudgpu_create_sampler(device, &sampler_desc, &sampler), "spudgpu_create_sampler");

	spudgpu_descriptor_set_layout_desc set_layout_desc = {
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT},
	        {.binding = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT},
	    },
	    .binding_count = 2,
	};
	spudgpu_descriptor_set_layout set_layout = NULL;
	check(spudgpu_create_descriptor_set_layout(device, &set_layout_desc, &set_layout), "spudgpu_create_descriptor_set_layout");

	spudgpu_descriptor_pool_desc pool_desc = {
	    .max_sets   = TEXTURE_COUNT,
	    .pool_sizes = {
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .count = TEXTURE_COUNT},
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = TEXTURE_COUNT},
	    },
	    .pool_size_count = 2,
	};
	spudgpu_descriptor_pool descriptor_pool = NULL;
	check(spudgpu_create_descriptor_pool(device, &pool_desc, &descriptor_pool), "spudgpu_create_descriptor_pool");

	spudgpu_descriptor_set texture_sets[TEXTURE_COUNT];
	for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
		spudgpu_descriptor_set_desc set_desc = {
		    .pool        = descriptor_pool,
		    .set_layouts = {set_layout},
		    .set_count   = 1,
		};
		check(spudgpu_create_descriptor_sets(device, &set_desc, &texture_sets[n]), "spudgpu_create_descriptor_sets");
	}
	write_texture_sets(device, &textures, texture_sets, sampler);
	set_title(window, &textures);

	spudgpu_shader_module vertex_module   = load_shader_module(device, "shaders/heaps.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	spudgpu_shader_module fragment_module = load_shader_module(device, "shaders/heaps.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);

	spudgpu_shader_pipeline_desc pipeline_desc = {
	    .vertex_module     = vertex_module,
	    .fragment_module   = fragment_module,
	    .vertex_attributes = {
	        {.location = 0, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, position)},
	        {.location = 1, .binding = 0, .format = SPUDGPU_FORMAT_R32G32_FLOAT, .offset = offsetof(Vertex, uv)},
	    },
	    .vertex_attribute_count = 2,
	    .vertex_bindings        = {
            {.binding = 0, .stride = sizeof(Vertex), .per_instance = false},
        },
	    .vertex_binding_count        = 1,
	    .primitive_topology          = SPUDGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
	    .cull_mode                   = SPUDGPU_CULL_MODE_NONE,
	    .front_face_ccw              = true,
	    .color_attachment_formats    = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count      = 1,
	    .depth_format                = SPUDGPU_FORMAT_UNKNOWN,
	    .descriptor_set_layouts      = {set_layout},
	    .descriptor_set_layout_count = 1,
	};
	spudgpu_shader_pipeline pipeline = NULL;
	check(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline), "spudgpu_create_shader_pipeline");

	bool running = true;
	while (running) {
		if (frame_limit != 0 && frames_run == frame_limit)
			break;
		++frames_run;
		bool switch_textures = false;
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
			if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_SPACE && !event.key.repeat)
				switch_textures = true;
		}

		// D3D12SmallResources::OnKeyDown: wait for the GPU, then make the
		// textures again the other way. The old ones (and their heap) go
		// first: nothing references them once the queue is idle.
		if (switch_textures) {
			spudgpu_queue_wait_idle(graphics_queue);
			const bool placed = !textures.placed;
			destroy_textures(&textures);
			create_textures(device, placed, &textures);

			spudgpu_reset_command_allocator(command_allocator);
			spudgpu_begin_command_list(cmd);
			record_texture_upload(cmd, &textures, staging, texture_offsets, row_pitch);
			spudgpu_end_command_list(cmd);
			spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
			spudgpu_queue_wait_idle(graphics_queue);

			write_texture_sets(device, &textures, texture_sets, sampler);
			set_title(window, &textures);
		}

		uint32_t image_index               = spudgpu_swap_chain_acquire_next_image(swap_chain);
		spudgpu_image_view backbuffer_view = spudgpu_get_swap_chain_image_view(swap_chain, image_index);

		spudgpu_reset_command_allocator(command_allocator);
		spudgpu_begin_command_list(cmd);

		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		spudgpu_rendering_begin_desc rendering_desc = {
		    .color_attachments = {
		        {
		            .image_view  = backbuffer_view,
		            .load_op     = SPUDGPU_LOAD_OP_CLEAR,
		            .store_op    = SPUDGPU_STORE_OP_STORE,
		            .clear_color = {0.0f, 0.2f, 0.4f, 1.0f},
		        },
		    },
		    .color_attachment_count = 1,
		    .width                  = WINDOW_WIDTH,
		    .height                 = WINDOW_HEIGHT,
		};
		spudgpu_cmd_begin_rendering(cmd, &rendering_desc);

		spudgpu_cmd_bind_pipeline(cmd, pipeline);

		SPUDGPU_VIEWPORT viewport = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT, .minDepth = 0.0f, .maxDepth = 1.0f};
		spudgpu_cmd_set_viewports(cmd, 0, 1, &viewport);
		SPUDGPU_SCISSOR_RECT scissor = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};
		spudgpu_cmd_set_scissor_rects(cmd, 0, 1, &scissor);

		spudgpu_cmd_set_vertex_buffers(cmd, 0, 1, &vertex_buffer_view);
		for (uint32_t n = 0; n < TEXTURE_COUNT; n++) {
			spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 0, &texture_sets[n], 1);
			spudgpu_cmd_draw(cmd, 4, n * 4);
		}

		spudgpu_cmd_end_rendering(cmd);
		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, SPUDGPU_IMAGE_LAYOUT_PRESENT_SRC);

		spudgpu_end_command_list(cmd);

		spudgpu_submit_command_lists_synced(graphics_queue, &cmd, 1, swap_chain);
		spudgpu_swap_chain_present(swap_chain);
	}

	spudgpu_queue_wait_idle(graphics_queue);

	// Placed resources before the heaps they are in.
	spudgpu_destroy_shader_pipeline(pipeline);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(vertex_module);
	spudgpu_destroy_descriptor_pool(descriptor_pool);
	spudgpu_destroy_descriptor_set_layout(set_layout);
	spudgpu_destroy_sampler(sampler);
	destroy_textures(&textures);
	spudgpu_destroy_buffer_view(vertex_buffer_view);
	spudgpu_destroy_buffer(vertex_buffer);
	spudgpu_destroy_heap(vertex_heap);
	spudgpu_destroy_buffer(staging);
	spudgpu_destroy_heap(staging_heap); // NULL when the staging buffer is its own allocation
	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_swap_chain(swap_chain);
	spudgpu_destroy_surface(surface);
	spudgpu_destroy_instance(instance);

	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
