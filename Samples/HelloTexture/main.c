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

// Generates a simple black/white checkerboard, matching the original
// sample's D3D12HelloTexture::GenerateTextureData exactly (8x8 cells).
static uint8_t *generate_checkerboard_texture(uint32_t width, uint32_t height) {
	const uint32_t pixel_size  = 4;
	const uint32_t row_pitch   = width * pixel_size;
	const uint32_t cell_pitch  = row_pitch >> 3;
	const uint32_t cell_height = width >> 3;
	const uint32_t texture_size = row_pitch * height;

	uint8_t *data = (uint8_t *) malloc(texture_size);
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
	// upload command list to complete.
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

	SDL_Window *window = SDL_CreateWindow("SpudGPU HelloTexture", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		return 1;
	}

	spudgpu_instance instance = NULL;
	if (SPUDFAIL(spudgpu_create_instance("SpudGPUHelloTexture", 1, "SpudGPUSamples", 1, &instance))) {
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

	// Vertex buffer: one hardcoded triangle with UVs, UPLOAD memory
	// written once -- same aspect-ratio-corrected shape as the original.
	float aspect_ratio = (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT;
	Vertex vertices[3] = {
	    {{0.0f, 0.25f * aspect_ratio, 0.0f}, {0.5f, 0.0f}},
	    {{0.25f, -0.25f * aspect_ratio, 0.0f}, {1.0f, 1.0f}},
	    {{-0.25f, -0.25f * aspect_ratio, 0.0f}, {0.0f, 1.0f}},
	};

	spudgpu_buffer_desc vertex_buffer_desc = {
	    .usage        = SPUDGPU_BUFFER_USAGE_VERTEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(vertices),
	};
	spudgpu_buffer vertex_buffer = NULL;
	if (SPUDFAIL(spudgpu_create_buffer(device, &vertex_buffer_desc, &vertex_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer failed\n");
		return 1;
	}

	void *vertex_mapped = NULL;
	spudgpu_map_buffer(vertex_buffer, 0, 0, &vertex_mapped);
	memcpy(vertex_mapped, vertices, sizeof(vertices));
	spudgpu_flush_buffer(vertex_buffer, 0, 0);
	spudgpu_unmap_buffer(vertex_buffer);

	spudgpu_buffer_view_desc vertex_buffer_view_desc = {
	    .parent_buffer             = vertex_buffer,
	    .offset_from_parent_buffer = 0,
	    .stride                    = sizeof(Vertex),
	    .size                      = sizeof(vertices),
	};
	spudgpu_buffer_view vertex_buffer_view = NULL;
	spudgpu_create_buffer_view(vertex_buffer, &vertex_buffer_view_desc, &vertex_buffer_view);

	// ------------------------------------------------------------------
	// Texture + sampler. Upload happens on `cmd`, submitted once and waited
	// on below before the render loop starts.
	// ------------------------------------------------------------------

	spudgpu_begin_command_list(cmd);

	uint8_t *checkerboard = generate_checkerboard_texture(TEXTURE_WIDTH, TEXTURE_HEIGHT);
	spudgpu_buffer staging_buffer = NULL;
	spudgpu_image texture = upload_rgba8_texture(device, cmd, checkerboard, TEXTURE_WIDTH, TEXTURE_HEIGHT, &staging_buffer);
	free(checkerboard);

	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);
	spudgpu_destroy_buffer(staging_buffer);

	spudgpu_image_view_desc texture_view_desc = {
	    .parent_image = texture,
	    .type         = SPUDGPU_IMAGE_VIEW_TYPE_2D,
	    .subresource_range = {.aspect_mask = 1 /* COLOR */, .base_mip_level = 0, .mip_level_count = 1, .base_array_layer = 0, .array_layer_count = 1},
	};
	spudgpu_image_view texture_view = NULL;
	spudgpu_create_image_view(texture, &texture_view_desc, &texture_view);

	spudgpu_sampler_desc sampler_desc = {
	    .mag_filter     = SPUDGPU_FILTER_NEAREST,
	    .min_filter     = SPUDGPU_FILTER_NEAREST,
	    .mipmap_filter  = SPUDGPU_FILTER_NEAREST,
	    .address_mode_u = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .address_mode_v = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .address_mode_w = SPUDGPU_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .max_lod        = 0.0f,
	    .max_anisotropy = 1.0f,
	};
	spudgpu_sampler sampler = NULL;
	if (SPUDFAIL(spudgpu_create_sampler(device, &sampler_desc, &sampler))) {
		fprintf(stderr, "spudgpu_create_sampler failed\n");
		return 1;
	}

	// Descriptor set layout: the texture at binding 0 (SAMPLED_IMAGE) and its
	// sampler at binding 1 (SAMPLER), both visible to the fragment stage only
	// -- see the file header comment for why this isn't one
	// COMBINED_IMAGE_SAMPLER binding.
	spudgpu_descriptor_set_layout_desc set_layout_desc = {
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT},
	        {.binding = 1, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_FRAGMENT},
	    },
	    .binding_count = 2,
	};
	spudgpu_descriptor_set_layout set_layout = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_set_layout(device, &set_layout_desc, &set_layout))) {
		fprintf(stderr, "spudgpu_create_descriptor_set_layout failed\n");
		return 1;
	}

	spudgpu_descriptor_pool_desc pool_desc = {
	    .max_sets   = 1,
	    .pool_sizes = {
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .count = 1},
	        {.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER, .count = 1},
	    },
	    .pool_size_count = 2,
	};
	spudgpu_descriptor_pool descriptor_pool = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_pool(device, &pool_desc, &descriptor_pool))) {
		fprintf(stderr, "spudgpu_create_descriptor_pool failed\n");
		return 1;
	}

	spudgpu_descriptor_set_desc set_desc = {
	    .pool        = descriptor_pool,
	    .set_layouts = {set_layout},
	    .set_count   = 1,
	};
	spudgpu_descriptor_set texture_set = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_sets(device, &set_desc, &texture_set))) {
		fprintf(stderr, "spudgpu_create_descriptor_sets failed\n");
		return 1;
	}

	spudgpu_descriptor_image_info texture_image_info = {
	    .image_view   = texture_view,
	    .image_layout = SPUDGPU_IMAGE_LAYOUT_SHADER_READ_ONLY,
	};
	spudgpu_write_descriptor_set texture_writes[2] = {
	    {
	        .dst_set          = texture_set,
	        .dst_binding      = 0,
	        .descriptor_count = 1,
	        .descriptor_type  = SPUDGPU_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
	        .image_info       = &texture_image_info,
	    },
	    {
	        .dst_set          = texture_set,
	        .dst_binding      = 1,
	        .descriptor_count = 1,
	        .descriptor_type  = SPUDGPU_DESCRIPTOR_TYPE_SAMPLER,
	        .sampler          = sampler,
	    },
	};
	spudgpu_update_descriptor_sets(device, texture_writes, 2);

	spudgpu_shader_module vertex_module   = load_shader_module(device, "shaders/texture.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	spudgpu_shader_module fragment_module = load_shader_module(device, "shaders/texture.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);

	spudgpu_shader_pipeline_desc pipeline_desc = {
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
	spudgpu_shader_pipeline pipeline = NULL;
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		return 1;
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
	spudgpu_destroy_instance(instance);

	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
