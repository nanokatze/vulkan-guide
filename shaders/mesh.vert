#version 450

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;

layout (location = 0) out vec3 outColor;

layout( push_constant ) uniform constants
{
layout(offset = 0) mat4 data;
} PushConstants;

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
	mat4 model = world.cameraMatrix* ubo.modelMatrix;//PushConstants.data;
	gl_Position = model * vec4(vPosition, 1.0f);	

	outColor = vNormal;
}

