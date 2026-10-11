//
// SpudGPU port of D3D12DepthBoundsTest from d3d12samples. Same scene (one
// hardcoded, unindexed triangle whose three vertices sit at different
// depths) and the same animated depth-bounds window, rebuilt against
// spudgpu_* calls instead of ID3D12*/DXGI -- see ../../README.md for the one
// deliberate deviation this port takes (a highlight-colored second pass
// instead of the original's colorless depth-only priming pass).
//

#include <SDL3/SDL.h>
#include <math.h>
#include <spudgpu.h>
#include <spudgpu_sdl3.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

// aspect_mask is a plain uint64_t on spudgpu_image_subresource_range (see
// spudgpu.h) -- SpudGPU has no enum/constant of its own for it yet, so this
// mirrors the raw VK_IMAGE_ASPECT_DEPTH_BIT value every backend agrees on,
// the same local #define SpudGPUDynamicIndexing already uses.
#define SPUDGPU_ASPECT_DEPTH_BIT 0x00000002u

typedef struct Vertex {
	float position[3];
	float color[4];
} Vertex;

static void *read_entire_file(const char *path, size_t *out_size) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open shader file: %s\n", path);
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

int main(void) {
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	SDL_Window *window = SDL_CreateWindow("SpudGPU Depth Bounds Test", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		return 1;
	}

	spudgpu_instance instance = NULL;
	if (SPUDFAIL(spudgpu_create_instance("SpudGPUDepthBoundsTest", 1, "SpudGPUSamples", 1, &instance))) {
		fprintf(stderr, "spudgpu_create_instance failed\n");
		return 1;
	}

	spudgpu_device *devices = NULL;
	uint32_t device_count   = 0;
	if (SPUDFAIL(spudgpu_enumerate_devices(instance, &devices, &device_count)) || device_count == 0) {
		fprintf(stderr, "spudgpu_enumerate_devices failed or returned no devices\n");
		return 1;
	}
	spudgpu_device device = devices[0];

	SPUDGPU_DEVICE_PROPERTIES device_properties = {0};
	spudgpu_get_device_properties(device, &device_properties);
	printf("Using device: %s\n", device_properties.description);

	spudgpu_depth_bounds_capabilities depth_bounds_caps = {0};
	spudgpu_get_depth_bounds_capabilities(device, &depth_bounds_caps);
	printf(
	    "Depth bounds test supported: %s\n",
	    depth_bounds_caps.supported ? "yes" : "no (bounds window will have no visible effect)");

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

	// One hardcoded, unindexed triangle whose three vertices sit at
	// different depths (0.1/0.9/0.5) -- same scene as the D3D12 original,
	// same aspect-ratio correction on the Y axis.
	const float aspect_ratio = (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT;
	const Vertex vertices[3] = {
	    {{0.00f, 0.25f * aspect_ratio, 0.1f}, {1.0f, 0.0f, 0.0f, 1.0f}}, // Top, red
	    {{0.25f, -0.25f * aspect_ratio, 0.9f}, {0.0f, 1.0f, 0.0f, 1.0f}}, // Right, green
	    {{-0.25f, -0.25f * aspect_ratio, 0.5f}, {0.0f, 0.0f, 1.0f, 1.0f}}, // Left, blue
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

	void *mapped = NULL;
	spudgpu_map_buffer(vertex_buffer, 0, 0, &mapped);
	memcpy(mapped, vertices, sizeof(vertices));
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

	spudgpu_shader_module vertex_module    = load_shader_module(device, "shaders/triangle.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	spudgpu_shader_module fragment_module  = load_shader_module(device, "shaders/triangle.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	spudgpu_shader_module highlight_module = load_shader_module(device, "shaders/highlight.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);

	// Priming pass: renders the triangle's true per-vertex colors and
	// writes its true per-vertex depth into the depth attachment. Ordinary
	// depth test, no depth bounds test.
	spudgpu_shader_pipeline_desc prime_pipeline_desc = {
	    .vertex_module   = vertex_module,
	    .fragment_module = fragment_module,
	    .vertex_attributes = {
	        {.location = 0, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, position)},
	        {.location = 1, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32A32_FLOAT, .offset = offsetof(Vertex, color)},
	    },
	    .vertex_attribute_count = 2,
	    .vertex_bindings        = {
	        {.binding = 0, .stride = sizeof(Vertex), .per_instance = false},
	    },
	    .vertex_binding_count   = 1,
	    .primitive_topology      = SPUDGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	    .cull_mode               = SPUDGPU_CULL_MODE_NONE,
	    .front_face_ccw          = true,
	    .depth_test_enable       = true,
	    .depth_write_enable      = true,
	    .depth_compare_op        = SPUDGPU_COMPARE_OP_LESS,
	    .color_attachment_format = SPUDGPU_FORMAT_B8G8R8A8_UNORM,
	    .depth_format            = SPUDGPU_FORMAT_D32_FLOAT,
	};
	spudgpu_shader_pipeline prime_pipeline = NULL;
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &prime_pipeline_desc, &prime_pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline (prime) failed\n");
		return 1;
	}

	// Depth-bounds pass: redraws the same triangle in a fixed highlight
	// color, with the ordinary depth test off and the depth bounds test on
	// instead -- a fragment is only visible here if the depth value the
	// priming pass just wrote at that pixel falls inside
	// spudgpu_cmd_set_depth_bounds' animated [min, max] window.
	spudgpu_shader_pipeline_desc bounds_pipeline_desc = prime_pipeline_desc;
	bounds_pipeline_desc.fragment_module         = highlight_module;
	bounds_pipeline_desc.depth_test_enable       = false;
	bounds_pipeline_desc.depth_write_enable      = false;
	bounds_pipeline_desc.depth_bounds_test_enable = true;
	spudgpu_shader_pipeline bounds_pipeline = NULL;
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &bounds_pipeline_desc, &bounds_pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline (bounds) failed\n");
		return 1;
	}

	spudgpu_command_allocator_desc allocator_desc = {.type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_command_allocator command_allocator   = NULL;
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_command_list cmd = NULL;
	spudgpu_create_command_list(command_allocator, &cmd);

	// Depth buffer, created and transitioned once up front -- it never
	// leaves SPUDGPU_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the
	// life of this sample.
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

	spudgpu_begin_command_list(cmd);
	spudgpu_cmd_image_barrier(cmd, depth_image, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);

	bool running          = true;
	uint64_t frame_number = 0;
	while (running) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
		}

		frame_number++;

		uint32_t image_index              = spudgpu_swap_chain_acquire_next_image(swap_chain);
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
		            .clear_color = {0.392f, 0.584f, 0.929f, 1.0f},
		        },
		    },
		    .color_attachment_count = 1,
		    .depth_attachment       = {.image_view = depth_view, .depth_load_op = SPUDGPU_LOAD_OP_CLEAR, .depth_store_op = SPUDGPU_STORE_OP_STORE, .clear_depth = 1.0f},
		    .width                  = WINDOW_WIDTH,
		    .height                 = WINDOW_HEIGHT,
		};
		spudgpu_cmd_begin_rendering(cmd, &rendering_desc);

		SPUDGPU_VIEWPORT viewport = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT, .minDepth = 0.0f, .maxDepth = 1.0f};
		spudgpu_cmd_set_viewports(cmd, 0, 1, &viewport);

		SPUDGPU_SCISSOR_RECT scissor = {.x = 0, .y = 0, .width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};
		spudgpu_cmd_set_scissor_rects(cmd, 0, 1, &scissor);

		spudgpu_cmd_set_vertex_buffers(cmd, 0, 1, &vertex_buffer_view);

		// Prime the depth attachment with the triangle's real per-vertex
		// depth values.
		spudgpu_cmd_bind_pipeline(cmd, prime_pipeline);
		spudgpu_cmd_draw(cmd, 3, 0);

		// Slide the bounds window back and forth across [0, 0.25] each way
		// from the middle -- matches the D3D12 original's sinf-driven
		// animation exactly.
		float f = 0.125f + sinf((float) (frame_number & 0x7F) / 127.0f) * 0.125f;
		spudgpu_cmd_set_depth_bounds(cmd, 0.0f + f, 1.0f - f);

		spudgpu_cmd_bind_pipeline(cmd, bounds_pipeline);
		spudgpu_cmd_draw(cmd, 3, 0);

		spudgpu_cmd_set_depth_bounds(cmd, 0.0f, 1.0f);

		spudgpu_cmd_end_rendering(cmd);
		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, SPUDGPU_IMAGE_LAYOUT_PRESENT_SRC);

		spudgpu_end_command_list(cmd);

		spudgpu_submit_command_lists_synced(graphics_queue, &cmd, 1, swap_chain);
		spudgpu_swap_chain_present(swap_chain);
	}

	spudgpu_queue_wait_idle(graphics_queue);

	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_image_view(depth_view);
	spudgpu_destroy_image(depth_image);
	spudgpu_destroy_shader_pipeline(bounds_pipeline);
	spudgpu_destroy_shader_pipeline(prime_pipeline);
	spudgpu_destroy_shader_module(highlight_module);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(vertex_module);
	spudgpu_destroy_buffer_view(vertex_buffer_view);
	spudgpu_destroy_buffer(vertex_buffer);
	spudgpu_destroy_swap_chain(swap_chain);
	spudgpu_destroy_surface(surface);
	spudgpu_destroy_instance(instance);

	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
