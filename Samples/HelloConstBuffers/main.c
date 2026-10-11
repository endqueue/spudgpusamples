//
// SpudGPU port of D3D12HelloWorld/HelloConstBuffers from d3d12samples.
// Same triangle as ../HelloTriangle, but the vertex shader now offsets the
// triangle by a value read from a uniform buffer bound through a real
// spudgpu_descriptor_set -- see ../../README.md.
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

// Matches shaders/scene.vert's SceneConstantBuffer block exactly (std140: a
// single vec4 needs no manual padding, unlike the 256-byte-aligned D3D12
// original -- that padding is a CBV root-descriptor alignment rule specific
// to D3D12, not a Vulkan/Metal requirement).
typedef struct SceneConstantBuffer {
	float offset[4];
} SceneConstantBuffer;

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

	SDL_Window *window = SDL_CreateWindow("SpudGPU HelloConstBuffers", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		return 1;
	}

	spudgpu_instance instance = NULL;
	if (SPUDFAIL(spudgpu_create_instance("SpudGPUHelloConstBuffers", 1, "SpudGPUSamples", 1, &instance))) {
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
	// same "just enough to draw a triangle" scope as HelloTriangle.
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

	// Scene constant buffer: one vec4 offset, updated every frame and kept
	// mapped for the app's lifetime -- same pattern as the D3D12 original
	// (map once, memcpy new contents in each frame, never unmap until exit).
	spudgpu_buffer_desc scene_cb_desc = {
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(SceneConstantBuffer),
	};
	spudgpu_buffer scene_cb = NULL;
	if (SPUDFAIL(spudgpu_create_buffer(device, &scene_cb_desc, &scene_cb))) {
		fprintf(stderr, "spudgpu_create_buffer (scene constant buffer) failed\n");
		return 1;
	}
	SceneConstantBuffer scene_cb_data = {.offset = {0.0f, 0.0f, 0.0f, 0.0f}};
	void *scene_cb_mapped             = NULL;
	spudgpu_map_buffer(scene_cb, 0, 0, &scene_cb_mapped);
	memcpy(scene_cb_mapped, &scene_cb_data, sizeof(scene_cb_data));
	spudgpu_flush_buffer(scene_cb, 0, sizeof(scene_cb_data));

	// Descriptor set layout: one uniform buffer at binding 0, visible to the
	// vertex stage only (the fragment shader never reads it).
	spudgpu_descriptor_set_layout_desc set_layout_desc = {
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_VERTEX},
	    },
	    .binding_count = 1,
	};
	spudgpu_descriptor_set_layout set_layout = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_set_layout(device, &set_layout_desc, &set_layout))) {
		fprintf(stderr, "spudgpu_create_descriptor_set_layout failed\n");
		return 1;
	}

	spudgpu_descriptor_pool_desc pool_desc = {
	    .max_sets       = 1,
	    .pool_sizes     = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1}},
	    .pool_size_count = 1,
	};
	spudgpu_descriptor_pool descriptor_pool = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_pool(device, &pool_desc, &descriptor_pool))) {
		fprintf(stderr, "spudgpu_create_descriptor_pool failed\n");
		return 1;
	}

	spudgpu_descriptor_set_desc set_desc = {
	    .pool       = descriptor_pool,
	    .set_layouts = {set_layout},
	    .set_count  = 1,
	};
	spudgpu_descriptor_set scene_set = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_sets(device, &set_desc, &scene_set))) {
		fprintf(stderr, "spudgpu_create_descriptor_sets failed\n");
		return 1;
	}

	spudgpu_descriptor_buffer_info scene_cb_info = {
	    .buffer = scene_cb,
	    .offset = 0,
	    .range  = sizeof(SceneConstantBuffer),
	};
	spudgpu_write_descriptor_set scene_write = {
	    .dst_set           = scene_set,
	    .dst_binding       = 0,
	    .dst_array_element = 0,
	    .descriptor_count  = 1,
	    .descriptor_type   = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .buffer_info       = &scene_cb_info,
	};
	spudgpu_update_descriptor_sets(device, &scene_write, 1);

	spudgpu_shader_module vertex_module   = load_shader_module(device, "shaders/scene.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	spudgpu_shader_module fragment_module = load_shader_module(device, "shaders/scene.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);

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
	    .descriptor_set_layouts     = {set_layout},
	    .descriptor_set_layout_count = 1,
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

	const float translation_speed = 0.005f;
	const float offset_bounds     = 1.25f;

	bool running = true;
	while (running) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT)
				running = false;
		}

		scene_cb_data.offset[0] += translation_speed;
		if (scene_cb_data.offset[0] > offset_bounds)
			scene_cb_data.offset[0] = -offset_bounds;
		memcpy(scene_cb_mapped, &scene_cb_data, sizeof(scene_cb_data));
		spudgpu_flush_buffer(scene_cb, 0, sizeof(scene_cb_data));

		uint32_t image_index              = spudgpu_swap_chain_acquire_next_image(swap_chain);
		spudgpu_image_view backbuffer_view = spudgpu_get_swap_chain_image_view(swap_chain, image_index);

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
		spudgpu_cmd_bind_descriptor_sets(cmd, pipeline, 0, &scene_set, 1);

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
	spudgpu_destroy_descriptor_pool(descriptor_pool);
	spudgpu_destroy_descriptor_set_layout(set_layout);
	spudgpu_unmap_buffer(scene_cb);
	spudgpu_destroy_buffer(scene_cb);
	spudgpu_destroy_buffer_view(vertex_buffer_view);
	spudgpu_destroy_buffer(vertex_buffer);
	spudgpu_destroy_swap_chain(swap_chain);
	spudgpu_destroy_surface(surface);
	spudgpu_destroy_instance(instance);

	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
