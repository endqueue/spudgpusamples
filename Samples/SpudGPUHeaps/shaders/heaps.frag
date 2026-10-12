#version 450

// D3D12SmallResources' PSMain: the quad's texture, sampled. Separate texture
// and sampler (not a combined sampler2D), for the reason ../../HelloTexture
// gives.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;

void main() {
    out_color = texture(sampler2D(tex, samp), frag_uv);
}
