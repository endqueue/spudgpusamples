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
	int exit_code                               = 1;
	SDL_Window *window                          = NULL;
	spudgpu_instance instance                   = NULL;
	spudgpu_device device                       = NULL;
	spudgpu_command_queue graphics_queue        = NULL;
	spudgpu_surface surface                     = NULL;
	spudgpu_swap_chain swap_chain               = NULL;
	spudgpu_buffer vertex_buffer                = NULL;
	spudgpu_buffer_view vertex_buffer_view      = NULL;
	spudgpu_shader_module vertex_module         = NULL;
	spudgpu_shader_module fragment_module       = NULL;
	spudgpu_shader_module highlight_module      = NULL;
	spudgpu_shader_pipeline prime_pipeline      = NULL;
	spudgpu_shader_pipeline bounds_pipeline     = NULL;
	spudgpu_command_allocator command_allocator = NULL;
	spudgpu_command_list cmd                    = NULL;
	spudgpu_image depth_image                   = NULL;
	spudgpu_image_view depth_view               = NULL;

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	window = SDL_CreateWindow("SpudGPU Depth Bounds Test", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		goto cleanup;
	}

	const SPUD_ALLOCATOR gpu_allocator           = spud_allocator_system();
	const spudgpu_instance_desc gpu_instance_desc = {
	    .base                = {SPUD_STRUCT_TYPE_GPU_INSTANCE_DESC, NULL},
	    .allocator           = &gpu_allocator,
	    .application_name    = "SpudGPUDepthBoundsTest",
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

	// The depth bounds test where the adapter has it; the sample runs
	// without it, and the window then has no visible effect.
	const bool depth_bounds = (adapter_properties.supported_features & SPUDGPU_FEATURE_DEPTH_BOUNDS_TEST) != 0;
	printf("Depth bounds test supported: %s\n", depth_bounds ? "yes" : "no (bounds window will have no visible effect)");

	// A window to present to and one direct queue, and exactly the features
	// the sample uses: a device has only what it is created with.
	const spudgpu_device_desc device_desc = {
	    .base               = {SPUD_STRUCT_TYPE_GPU_DEVICE_DESC, NULL},
	    .required_features  = SPUDGPU_FEATURE_PRESENT | (adapter_properties.supported_features & SPUDGPU_FEATURE_DEPTH_BOUNDS_TEST),
	    .direct_queue_count = 1,
	    // What the sample's descriptor pools hold together, with room to
	    // spare: the caller's to choose, and refused past the adapter's
	    // limit.
	    .max_descriptors    = 1024,
	    .max_samplers       = 64,
	    .debug_name         = "SpudGPUDepthBoundsTest",
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
	    // The window can't be resized and the back buffers are its size, so
	    // nothing is scaled; ONE_TO_ONE is the scaling every surface takes.
	    .present_scaling = SPUDGPU_PRESENT_SCALING_ONE_TO_ONE,
	    .fullscreen_mode = SPUDGPU_FULLSCREEN_MODE_WINDOWED,
	};
	if (SPUDFAIL(spudgpu_create_swap_chain(device, &swap_chain_desc, &swap_chain))) {
		fprintf(stderr, "spudgpu_create_swap_chain failed\n");
		goto cleanup;
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
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_VERTEX,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(vertices),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &vertex_buffer_desc, &vertex_buffer))) {
		fprintf(stderr, "spudgpu_create_buffer failed\n");
		goto cleanup;
	}

	void *mapped = NULL;
	spudgpu_map_buffer(vertex_buffer, 0, 0, &mapped);
	memcpy(mapped, vertices, sizeof(vertices));
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

	vertex_module    = load_shader_module(device, "shaders/triangle.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	fragment_module  = load_shader_module(device, "shaders/triangle.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	highlight_module = load_shader_module(device, "shaders/highlight.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	if (!vertex_module || !fragment_module || !highlight_module)
		goto cleanup;

	// Priming pass: renders the triangle's true per-vertex colors and
	// writes its true per-vertex depth into the depth attachment. Ordinary
	// depth test, no depth bounds test.
	spudgpu_shader_pipeline_desc prime_pipeline_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SHADER_PIPELINE_DESC, NULL},
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
	    .color_attachment_formats = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count  = 1,
	    .depth_format            = SPUDGPU_FORMAT_D32_FLOAT,
	};
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &prime_pipeline_desc, &prime_pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline (prime) failed\n");
		goto cleanup;
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
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &bounds_pipeline_desc, &bounds_pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline (bounds) failed\n");
		goto cleanup;
	}

	spudgpu_command_allocator_desc allocator_desc = {.base = {SPUD_STRUCT_TYPE_GPU_COMMAND_ALLOCATOR_DESC, NULL}, .type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_create_command_list(command_allocator, &cmd);

	// Depth buffer, created and transitioned once up front -- it never
	// leaves SPUDGPU_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the
	// life of this sample.
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

	spudgpu_begin_command_list(cmd);
	spudgpu_cmd_image_barrier(cmd, depth_image, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	spudgpu_end_command_list(cmd);
	spudgpu_submit_command_lists(graphics_queue, &cmd, 1);
	spudgpu_queue_wait_idle(graphics_queue);

	bool running          = true;
	uint64_t frame_number = 0;
	while (running) {
		if (frame_limit != 0 && frames_run == frame_limit)
			break;
		++frames_run;
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
		    .base = {SPUD_STRUCT_TYPE_GPU_RENDERING_BEGIN_DESC, NULL},
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
		if (depth_bounds)
			spudgpu_cmd_set_depth_bounds(cmd, 0.0f + f, 1.0f - f);

		spudgpu_cmd_bind_pipeline(cmd, bounds_pipeline);
		spudgpu_cmd_draw(cmd, 3, 0);

		if (depth_bounds)
			spudgpu_cmd_set_depth_bounds(cmd, 0.0f, 1.0f);

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
	spudgpu_destroy_device(device);
	spudgpu_destroy_instance(instance);

	if (window)
		SDL_DestroyWindow(window);
	SDL_Quit();
	return exit_code;
}
