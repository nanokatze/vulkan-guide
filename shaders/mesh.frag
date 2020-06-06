#version 450

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform WorldParameters {
    mat4 cameraMatrix;   
	vec4 ambientColor;
} world;

layout(set = 1, binding = 0) uniform ObjectUniforms {
    mat4 modelMatrix;   
	vec4 shineColor;
} ubo;


void main() 
{
	outFragColor = vec4(inColor,1.0f);
}

