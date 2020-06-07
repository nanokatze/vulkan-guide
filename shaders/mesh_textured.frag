#version 450

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform WorldParameters {
    mat4 cameraMatrix;   
	vec4 ambientColor;
} world;

layout(set = 1, binding = 0) uniform ObjectUniforms {
    mat4 modelMatrix;   
	vec4 shineColor;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D tex1;

void main() 
{
	vec3 color = texture(tex1,inUV).xyz;
	outFragColor = vec4(color,1.0f);
}

