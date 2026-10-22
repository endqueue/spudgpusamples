//
// SpudGPU port of D3D12HelloWorld/HelloTexture from d3d12samples. Same
// unindexed triangle as ../HelloTriangle, but the pixel shader now samples a
// procedurally-generated black/white checkerboard texture instead of
// interpolating per-vertex color -- see ../../README.md.
//
// Three deliberate deviations from the original:
// - No static/immutable root-signature sampler. SpudGPU has no equivalent to
//   D3D12's D3D12_STATIC_SAMPLER_DESC / Vulkan's immutable samplers yet (see
//   spudlib/CLAUDE.md's "Known gaps") -- this port uses a real, dynamic
//   spudgpu_sampler instead, which is functionally identical for a sampler
//   that never changes for the life of the app, just not the zero-descriptor-
//   cost mechanism D3D12's static sampler is.
// - Separate SAMPLED_IMAGE + SAMPLER descriptors instead of one
//   COMBINED_IMAGE_SAMPLER. The combined type exists in spudgpu.h and works
//   on Vulkan/D3D12, but its Metal cross-compile fails below Metal 3:
//   spudgpumetaldescriptors.m already documents that the sampler half of a
//   COMBINED_IMAGE_SAMPLER is "intentionally not written" there, and
//   SPIRV-Cross confirms why -- under SpudGPU's per-set-argument-buffer
//   scheme, a GLSL sampler2D's synthesized sampler member overlaps the
//   texture's own binding slot, which SPIRV-Cross only allows via "full
//   mutable aliasing" on Metal 3+. This port instead uses the same split
//   texture2D/sampler idiom SpudGPUDynamicIndexing's bindless design already
//   established, which cross-compiles cleanly on all three backends.
// - CLAMP_TO_EDGE addressing instead of the original's CLAMP_TO_BORDER with a
//   transparent-black border color. spudgpu_sampler_desc has no border-color
//   field. This triangle's UVs ((0.5,0), (1,1), (0,1)) never leave [0,1], so
//   the two addressing modes are visually identical here regardless.
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

#define TEXTURE_WIDTH 256
#define TEXTURE_HEIGHT 256

typedef struct Vertex {
	float position[3];
	float uv[2];
} Vertex;

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

// Generates a simple black/white checkerboard, matching the original
// sample's D3D12HelloTexture::GenerateTextureData exactly (8x8 cells).
// NULL, printed, if there's no memory for it.
static uint8_t *generate_checkerboard_texture(uint32_t width, uint32_t height) {
	const uint32_t pixel_size  = 4;
	const uint32_t row_pitch   = width * pixel_size;
	const uint32_t cell_pitch  = row_pitch >> 3;
	const uint32_t cell_height = width >> 3;
	const uint32_t texture_size = row_pitch * height;

	uint8_t *data = (uint8_t *) malloc(texture_size);
	if (!data) {
		fprintf(stderr, "No memory for checkerboard texture\n");
		return NULL;
	}
	for (uint32_t n = 0; n < texture_size; n += pixel_size) {
		uint32_t x = n % row_pitch;
		uint32_t y = n / row_pitch;
		uint32_t i = x / cell_pitch;
		uint32_t j = y / cell_height;

		uint8_t value  = (i % 2 == j % 2) ? 0x00 : 0xff;
		data[n]     = value;
		data[n + 1] = value;
		data[n + 2] = value;
		data[n + 3] = 0xff;
	}
	return data;
}

// Uploads `pixels` (tightly packed, width*height*4 bytes) into a freshly
// created SPUDGPU_FORMAT_R8G8B8A8_UNORM image, via a staging buffer sized
// and strided exactly as spudgpu_get_image_buffer_copy_size reports --
// portable across Vulkan (no alignment requirement) and D3D12 (256-byte row
// pitch alignment), rather than assuming a tightly-packed row pitch.
// NULL, printed, on failure.
static spudgpu_image upload_rgba8_texture(
    spudgpu_device device,
    spudgpu_command_list cmd,
    const uint8_t *pixels,
    uint32_t width,
    uint32_t height,
    spudgpu_buffer *out_staging_buffer) {
	spudgpu_image_desc image_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_IMAGE_DESC, NULL},
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
		return NULL;
	}

	uint64_t row_pitch = 0, total_size = 0;
	spudgpu_get_image_buffer_copy_size(image, 0, &row_pitch, &total_size);
	if (row_pitch == 0) row_pitch = (uint64_t) width * 4;
	if (total_size == 0) total_size = row_pitch * height;

	spudgpu_buffer_desc staging_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_TRANSFER_SRC,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = total_size,
	};
	spudgpu_buffer staging = NULL;
	if (SPUDFAIL(spudgpu_create_buffer(device, &staging_desc, &staging))) {
		fprintf(stderr, "spudgpu_create_buffer (staging) failed\n");
		spudgpu_destroy_image(image);
		return NULL;
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
	// upload command list to complete.
	*out_staging_buffer = staging;
	return image;
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
	spudgpu_image texture                       = NULL;
	spudgpu_image_view texture_view             = NULL;
	spudgpu_sampler sampler                     = NULL;
	spudgpu_descriptor_set_layout set_layout    = NULL;
	spudgpu_descriptor_pool descriptor_pool     = NULL;
	spudgpu_shader_module vertex_module         = NULL;
	spudgpu_shader_module fragment_module       = NULL;
	spudgpu_shader_pipeline pipeline            = NULL;

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	window = SDL_CreateWindow("SpudGPU HelloTexture", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		goto cleanup;
	}

	const SPUD_ALLOCATOR gpu_allocator           = spud_allocator_system();
	const spudgpu_instance_desc gpu_instance_desc = {
	    .base                = {SPUD_STRUCT_TYPE_GPU_INSTANCE_DESC, NULL},
	    .allocator           = &gpu_allocator,
	    .application_name    = "SpudGPUHelloTexture",
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
	    .debug_name         = "HelloTexture",
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
	    // Metal's CAMetalLayer hands out one drawable at a time -- no stable
	    // N-image array to double-buffer against like Vulkan/D3D12, so
	    // spudgpu_create_swap_chain rejects anything but 1 on that backend.
#if SPUDGPU_COMPILE_METAL_API
	    .buffer_count    = 1,
#else
	    .buffer_count    = 2,
#endif
	    .format          = SPUDGPU_FORMAT_B8G8R8A8_UNORM,
	    .present_mode    = SPUDGPU_PRESENT_MODE_FIFO,
	    .fullscreen_mode = SPUDGPU_FULLSCREEN_MODE_WINDOWED,
	};
	if (SPUDFAIL(spudgpu_create_swap_chain(device, &swap_chain_desc, &swap_chain))) {
		fprintf(stderr, "spudgpu_create_swap_chain failed\n");
		goto cleanup;
	}

	spudgpu_command_allocator_desc allocator_desc = {.base = {SPUD_STRUCT_TYPE_GPU_COMMAND_ALLOCATOR_DESC, NULL}, .type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_create_command_list(command_allocator, &cmd);

	// Vertex buffer: one hardcoded triangle with UVs, UPLOAD memory
	// written once -- same aspect-ratio-corrected shape as the original.
	float aspect_ratio = (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT;
	Vertex vertices[3] = {
	    {{0.0f, 0.25f * aspect_ratio, 0.0f}, {0.5f, 0.0f}},
	    {{0.25f, -0.25f * aspect_ratio, 0.0f}, {1.0f, 1.0f}},
	    {{-0.25f, -0.25f * aspect_ratio, 0.0f}, {0.0f, 1.0f}},
	};

	spudgpu_buffer_desc vertex_buffer_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_VERTEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(vertices),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &vertex_buffer_desc, &vertex_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer failed\n");
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

	// ------------------------------------------------------------------
	// Texture + sampler. Upload happens on `cmd`, submitted once and waited
	// on below before the render loop starts.
	// ------------------------------------------------------------------

	spudgpu_begin_command_list(cmd);

	uint8_t *checkerboard = generate_checkerboard_texture(TEXTURE_WIDTH, TEXTURE_HEIGHT);
	if (!checkerboard)
		goto cleanup;
	spudgpu_buffer staging_buffer = NULL;
	texture = upload_rgba8_texture(device, cmd, checkerboard, TEXTURE_WIDTH, TEXTURE_HEIGHT, &staging_buffer);
	free(checkerboard);
	if (!texture)
		goto cleanup;

	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);
	spudgpu_destroy_buffer(staging_buffer);

	spudgpu_image_view_desc texture_view_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_IMAGE_VIEW_DESC, NULL},
	    .parent_image = texture,
	    .type         = SPUDGPU_IMAGE_VIEW_TYPE_2D,
	    .subresource_range = {.aspect_mask = 1 /* COLOR */, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
	};
	spudgpu_create_image_view(texture, &texture_view_desc, &texture_view);

	spudgpu_sampler_desc sampler_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SAMPLER_DESC, NULL},
	    .mag_filter     = SPUDGPU_FILTER_NEAREST,
	    .min_filter     = SPUDGPU_FILTER_NEAREST,
	    .mipmap_filter  = SPUDGPU_FILTER_NEAREST,
	    .address_mode_u = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .address_mode_v = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .address_mode_w = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .max_lod        = 0.0f,
	    .max_anisotropy = 1.0f,
	};
	if (SPUDFAIL(spudgpu_create_sampler(device, &sampler_desc, &sampler))) {
		fprintf(stderr, "spudgpu_create_sampler failed\n");
		goto cleanup;
	}

	// Descriptor set layout: the texture at binding 0 (SAMPLED_IMAGE) and its
	// sampler at binding 1 (SAMPLER), both visible to the fragment stage only
	// -- see the file header comment for why this isn't one
	// COMBINED_IMAGE_SAMPLER binding.
	spudgpu_descriptor_set_layout_desc set_layout_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_LAYOUT_DESC, NULL},
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT},
	        {.binding = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT},
	    },
	    .binding_count = 2,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_set_layout(device, &set_layout_desc, &set_layout))) {
		fprintf(stderr, "spudgpu_create_descriptor_set_layout failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_pool_desc pool_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_POOL_DESC, NULL},
	    .max_sets   = 1,
	    .pool_sizes = {
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .count = 1},
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = 1},
	    },
	    .pool_size_count = 2,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_pool(device, &pool_desc, &descriptor_pool))) {
		fprintf(stderr, "spudgpu_create_descriptor_pool failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_set_desc set_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_DESC, NULL},
	    .pool        = descriptor_pool,
	    .set_layouts = {set_layout},
	    .set_count   = 1,
	};
	spudgpu_descriptor_set texture_set = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_sets(device, &set_desc, &texture_set))) {
		fprintf(stderr, "spudgpu_create_descriptor_sets failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_image_info texture_image_info = {
	    .image_view   = texture_view,
	    .image_layout = SPUDGPU_IMAGE_LAYOUT_SHADER_READ_ONLY,
	};
	spudgpu_write_descriptor_set texture_writes[2] = {
	    {
	        .base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL},
	        .dst_set          = texture_set,
	        .dst_binding      = 0,
	        .descriptor_count = 1,
	        .descriptor_type  = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
	        .image_info       = &texture_image_info,
	    },
	    {
	        .base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL},
	        .dst_set          = texture_set,
	        .dst_binding      = 1,
	        .descriptor_count = 1,
	        .descriptor_type  = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER,
	        .sampler          = sampler,
	    },
	};
	spudgpu_update_descriptor_sets(device, texture_writes, 2);

	vertex_module   = load_shader_module(device, "shaders/texture.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	fragment_module = load_shader_module(device, "shaders/texture.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	if (!vertex_module || !fragment_module)
		goto cleanup;

	spudgpu_shader_pipeline_desc pipeline_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SHADER_PIPELINE_DESC, NULL},
	    .vertex_module   = vertex_module,
	    .fragment_module = fragment_module,
	    .vertex_attributes = {
	        {.location = 0, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, position)},
	        {.location = 1, .binding = 0, .format = SPUDGPU_FORMAT_R32G32_FLOAT, .offset = offsetof(Vertex, uv)},
	    },
	    .vertex_attribute_count = 2,
	    .vertex_bindings        = {
	        {.binding = 0, .stride = sizeof(Vertex), .per_instance = false},
	    },
	    .vertex_binding_count    = 1,
	    .primitive_topology      = SPUDGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	    .cull_mode               = SPUDGPU_CULL_MODE_NONE,
	    .front_face_ccw          = true,
	    .color_attachment_formats = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count  = 1,
	    .depth_format            = SPUDGPU_FORMAT_UNKNOWN,
	    .descriptor_set_layouts      = {set_layout},
	    .descriptor_set_layout_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		goto cleanup;
	}

	bool running = true;
	while (running) {
		if (frame_limit != 0 && frames_run == frame_limit)
			break;
		++frames_run;
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
		}

		uint32_t image_index               = spudgpu_swap_chain_acquire_next_image(swap_chain);
		spudgpu_image_view backbuffer_view = spudgpu_get_swap_chain_image_view(swap_chain, image_index);

		spudgpu_reset_command_allocator(command_allocator);
		spudgpu_begin_command_list(cmd);

		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		spudgpu_rendering_begin_desc rendering_desc = {
		    .base = {SPUD_STRUCT_TYPE_GPU_RENDERING_BEGIN_DESC, NULL},
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
		spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 0, &texture_set, 1);

		SPUDGPU_VIEWPORT viewport = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT, .minDepth = 0.0f, .maxDepth = 1.0f};
		spudgpu_cmd_set_viewports(cmd, 0, 1, &viewport);

		SPUDGPU_SCISSOR_RECT scissor = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};
		spudgpu_cmd_set_scissor_rects(cmd, 0, 1, &scissor);

		spudgpu_cmd_set_vertex_buffers(cmd, 0, 1, &vertex_buffer_view);
		spudgpu_cmd_draw(cmd, 3, 0);

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

	spudgpu_destroy_shader_pipeline(pipeline);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(vertex_module);
	spudgpu_destroy_descriptor_pool(descriptor_pool);
	spudgpu_destroy_descriptor_set_layout(set_layout);
	spudgpu_destroy_sampler(sampler);
	spudgpu_destroy_image_view(texture_view);
	spudgpu_destroy_image(texture);
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
