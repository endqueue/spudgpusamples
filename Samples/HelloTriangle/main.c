//
// SpudGPU port of D3D12HelloWorld/HelloTriangle from d3d12samples.
// Same scene (one hardcoded, unindexed, unlit triangle), rebuilt entirely
// against spudgpu_* calls instead of ID3D12*/DXGI -- see ../../README.md.
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

typedef struct Vertex {
	float position[2];
	float color[3];
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

	SDL_Window *window = SDL_CreateWindow("SpudGPU HelloTriangle", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		return 1;
	}

	spudgpu_instance instance = NULL;
	if (SPUDFAIL(spudgpu_create_instance("SpudGPUHelloTriangle", 1, "SpudGPUSamples", 1, &instance))) {
		fprintf(stderr, "spudgpu_create_instance failed\n");
		return 1;
	}

	spudgpu_device *devices  = NULL;
	uint32_t device_count    = 0;
	if (SPUDFAIL(spudgpu_enumerate_devices(instance, &devices, &device_count)) || device_count == 0) {
		fprintf(stderr, "spudgpu_enumerate_devices failed or returned no devices\n");
		return 1;
	}
	spudgpu_device device = devices[0];

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

	// Vertex buffer: one hardcoded triangle, UPLOAD memory written once --
	// same "just enough to draw a triangle" scope as the D3D12 sample.
	static const Vertex vertices[3] = {
	    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
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

	spudgpu_shader_module vertex_module   = load_shader_module(device, "shaders/triangle.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	spudgpu_shader_module fragment_module = load_shader_module(device, "shaders/triangle.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);

	spudgpu_shader_pipeline_desc pipeline_desc = {
	    .vertex_module   = vertex_module,
	    .fragment_module = fragment_module,
	    .vertex_attributes = {
	        {.location = 0, .binding = 0, .format = SPUDGPU_FORMAT_R32G32_FLOAT, .offset = offsetof(Vertex, position)},
	        {.location = 1, .binding = 0, .format = SPUDGPU_FORMAT_R32G32B32_FLOAT, .offset = offsetof(Vertex, color)},
	    },
	    .vertex_attribute_count = 2,
	    .vertex_bindings        = {
	        {.binding = 0, .stride = sizeof(Vertex), .per_instance = false},
        },
	    .vertex_binding_count   = 1,
	    .primitive_topology     = SPUDGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	    .cull_mode              = SPUDGPU_CULL_MODE_NONE,
	    .front_face_ccw         = true,
	    .color_attachment_format = SPUDGPU_FORMAT_B8G8R8A8_UNORM,
	    .depth_format            = SPUDGPU_FORMAT_UNKNOWN,
	};
	spudgpu_shader_pipeline pipeline = NULL;
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		return 1;
	}

	spudgpu_command_allocator_desc allocator_desc = {.type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_command_allocator command_allocator   = NULL;
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_command_list cmd = NULL;
	spudgpu_create_command_list(command_allocator, &cmd);

	bool running = true;
	while (running) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
		}

		uint32_t image_index               = spudgpu_swap_chain_acquire_next_image(swap_chain);
		spudgpu_image_view backbuffer_view  = spudgpu_get_swap_chain_image_view(swap_chain, image_index);

		spudgpu_reset_command_allocator(command_allocator);
		spudgpu_begin_command_list(cmd);

		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_UNDEFINED, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		spudgpu_rendering_begin_desc rendering_desc = {
		    .color_attachments = {
		        {
		            .image_view = backbuffer_view,
		            .load_op    = SPUDGPU_LOAD_OP_CLEAR,
		            .store_op   = SPUDGPU_STORE_OP_STORE,
		            .clear_color = {0.02f, 0.02f, 0.05f, 1.0f},
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
		spudgpu_cmd_draw(cmd, 3, 0);

		spudgpu_cmd_end_rendering(cmd);
		spudgpu_cmd_image_barrier_view(cmd, backbuffer_view, SPUDGPU_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, SPUDGPU_IMAGE_LAYOUT_PRESENT_SRC);

		spudgpu_end_command_list(cmd);

		spudgpu_submit_command_lists_synced(graphics_queue, &cmd, 1, swap_chain);
		spudgpu_swap_chain_present(swap_chain);
	}

	spudgpu_queue_wait_idle(graphics_queue);

	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_shader_pipeline(pipeline);
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
