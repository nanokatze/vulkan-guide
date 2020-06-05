#version 450

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

layout( push_constant ) uniform constants
{
vec4 data;
} PushConstants;

void main() 
{
	outFragColor = vec4(inColor,1.0f);
}

