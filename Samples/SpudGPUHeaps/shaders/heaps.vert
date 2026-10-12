#version 450

// D3D12SmallResources' VSMain: the quad's position as given, its UV passed on.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 frag_uv;

void main() {
    gl_Position = vec4(in_position, 1.0);
    frag_uv = in_uv;
}
