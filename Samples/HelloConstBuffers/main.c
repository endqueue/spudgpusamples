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
	spudgpu_buffer scene_cb                     = NULL;
	spudgpu_descriptor_set_layout set_layout    = NULL;
	spudgpu_descriptor_pool descriptor_pool     = NULL;
	spudgpu_shader_module vertex_module         = NULL;
	spudgpu_shader_module fragment_module       = NULL;
	spudgpu_shader_pipeline pipeline            = NULL;
	spudgpu_command_allocator command_allocator = NULL;
	spudgpu_command_list cmd                    = NULL;

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	window = SDL_CreateWindow("SpudGPU HelloConstBuffers", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		goto cleanup;
	}

	const SPUD_ALLOCATOR gpu_allocator           = spud_allocator_system();
	const spudgpu_instance_desc gpu_instance_desc = {
	    .base                = {SPUD_STRUCT_TYPE_GPU_INSTANCE_DESC, NULL},
	    .allocator           = &gpu_allocator,
	    .application_name    = "SpudGPUHelloConstBuffers",
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
	    .debug_name         = "HelloConstBuffers",
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

	// Vertex buffer: one hardcoded triangle, UPLOAD memory written once --
	// same "just enough to draw a triangle" scope as HelloTriangle.
	static const Vertex vertices[3] = {
	    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
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

	// Scene constant buffer: one vec4 offset, updated every frame and kept
	// mapped for the app's lifetime -- same pattern as the D3D12 original
	// (map once, memcpy new contents in each frame, never unmap until exit).
	spudgpu_buffer_desc scene_cb_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_BUFFER_DESC, NULL},
	    .usage        = SPUDGPU_BUFFER_USAGE_UNIFORM,
	    .memory_kind  = SPUDGPU_MEMORY_KIND_UPLOAD,
	    .size         = sizeof(SceneConstantBuffer),
	};
	if (SPUDFAIL(spudgpu_create_buffer(device, &scene_cb_desc, &scene_cb))) {
		fprintf(stderr, "spudgpu_create_buffer (scene constant buffer) failed\n");
		goto cleanup;
	}
	SceneConstantBuffer scene_cb_data = {.offset = {0.0f, 0.0f, 0.0f, 0.0f}};
	void *scene_cb_mapped             = NULL;
	spudgpu_map_buffer(scene_cb, 0, 0, &scene_cb_mapped);
	memcpy(scene_cb_mapped, &scene_cb_data, sizeof(scene_cb_data));
	spudgpu_flush_buffer(scene_cb, 0, sizeof(scene_cb_data));

	// Descriptor set layout: one uniform buffer at binding 0, visible to the
	// vertex stage only (the fragment shader never reads it).
	spudgpu_descriptor_set_layout_desc set_layout_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_LAYOUT_DESC, NULL},
	    .bindings = {
	        {.binding = 0, .descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stage_flags = SPUDGPU_SHADER_STAGE_VERTEX},
	    },
	    .binding_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_set_layout(device, &set_layout_desc, &set_layout))) {
		fprintf(stderr, "spudgpu_create_descriptor_set_layout failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_pool_desc pool_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_POOL_DESC, NULL},
	    .max_sets       = 1,
	    .pool_sizes     = {{.descriptor_type = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1}},
	    .pool_size_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_descriptor_pool(device, &pool_desc, &descriptor_pool))) {
		fprintf(stderr, "spudgpu_create_descriptor_pool failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_set_desc set_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_DESCRIPTOR_SET_DESC, NULL},
	    .pool       = descriptor_pool,
	    .set_layouts = {set_layout},
	    .set_count  = 1,
	};
	spudgpu_descriptor_set scene_set = NULL;
	if (SPUDFAIL(spudgpu_create_descriptor_sets(device, &set_desc, &scene_set))) {
		fprintf(stderr, "spudgpu_create_descriptor_sets failed\n");
		goto cleanup;
	}

	spudgpu_descriptor_buffer_info scene_cb_info = {
	    .buffer = scene_cb,
	    .offset = 0,
	    .range  = sizeof(SceneConstantBuffer),
	};
	spudgpu_write_descriptor_set scene_write = {
	    .base = {SPUD_STRUCT_TYPE_GPU_WRITE_DESCRIPTOR_SET, NULL},
	    .dst_set           = scene_set,
	    .dst_binding       = 0,
	    .dst_array_element = 0,
	    .descriptor_count  = 1,
	    .descriptor_type   = SPUDGPU_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .buffer_info       = &scene_cb_info,
	};
	spudgpu_update_descriptor_sets(device, &scene_write, 1);

	vertex_module   = load_shader_module(device, "shaders/scene.vert.spv", SPUDGPU_SHADER_STAGE_VERTEX);
	fragment_module = load_shader_module(device, "shaders/scene.frag.spv", SPUDGPU_SHADER_STAGE_FRAGMENT);
	if (!vertex_module || !fragment_module)
		goto cleanup;

	spudgpu_shader_pipeline_desc pipeline_desc = {
	    .base = {SPUD_STRUCT_TYPE_GPU_SHADER_PIPELINE_DESC, NULL},
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
	    .color_attachment_formats = {SPUDGPU_FORMAT_B8G8R8A8_UNORM},
	    .color_attachment_count  = 1,
	    .depth_format            = SPUDGPU_FORMAT_UNKNOWN,
	    .descriptor_set_layouts     = {set_layout},
	    .descriptor_set_layout_count = 1,
	};
	if (SPUDFAIL(spudgpu_create_shader_pipeline(device, &pipeline_desc, &pipeline))) {
		fprintf(stderr, "spudgpu_create_shader_pipeline failed\n");
		goto cleanup;
	}

	spudgpu_command_allocator_desc allocator_desc = {.base = {SPUD_STRUCT_TYPE_GPU_COMMAND_ALLOCATOR_DESC, NULL}, .type = SPUDGPU_COMMAND_LIST_TYPE_DIRECT};
	spudgpu_create_command_allocator(device, &allocator_desc, &command_allocator);

	spudgpu_create_command_list(command_allocator, &cmd);

	const float translation_speed = 0.005f;
	const float offset_bounds     = 1.25f;

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
		    .base = {SPUD_STRUCT_TYPE_GPU_RENDERING_BEGIN_DESC, NULL},
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

	exit_code = 0;

cleanup:
	if (graphics_queue)
		spudgpu_queue_wait_idle(graphics_queue);

	spudgpu_destroy_command_list(cmd);
	spudgpu_destroy_command_allocator(command_allocator);
	spudgpu_destroy_shader_pipeline(pipeline);
	spudgpu_destroy_shader_module(fragment_module);
	spudgpu_destroy_shader_module(vertex_module);
	spudgpu_destroy_descriptor_pool(descriptor_pool);
	spudgpu_destroy_descriptor_set_layout(set_layout);
	if (scene_cb)
		spudgpu_unmap_buffer(scene_cb);
	spudgpu_destroy_buffer(scene_cb);
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
