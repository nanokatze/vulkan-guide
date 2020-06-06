#include "VkBootstrap.h"

#include "vulkan_guide.h"
#include <vector>
#include <array>
#include <fstream>
#include <functional>
#include <deque>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_obj_loader.h>


#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

//we want to immediately abort when there is an error. In normal engines this would give an error message to the user, or perform a dump of state.
using namespace std;
#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			std::cout <<"Detected Vulkan error: " << err << std::endl; \
			abort();                                                \
		}                                                           \
	} while (0)

//set to false to disable validation layers
const bool bUseValidationLayers = true;

namespace vkinit {

	VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule module) {
		VkPipelineShaderStageCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		info.pNext = nullptr;

		info.stage = stage;
		info.module = module;
		info.pName = "main";
		return info;
	}

	VkPipelineLayoutCreateInfo pipeline_layout_create_info() {
		VkPipelineLayoutCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		info.pNext = nullptr;

		//empty defaults
		info.flags = 0;
		info.setLayoutCount = 0;
		info.pSetLayouts = nullptr;
		info.pushConstantRangeCount = 0;
		info.pPushConstantRanges = nullptr;
		return info;
	}

	VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolResetFlags flags = 0) {
		VkCommandPoolCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		info.pNext = nullptr;

		info.flags = flags;
		return info;
	}
	VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.pNext = nullptr;

		info.commandPool = pool;
		info.commandBufferCount = count;
		info.level = level;
		return info;
	}

	VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags = 0) {
		VkCommandBufferBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		info.pNext = nullptr;

		info.pInheritanceInfo = nullptr;
		info.flags = flags;
		return info;
	}
	VkFramebufferCreateInfo framebuffer_create_info(VkRenderPass renderPass,VkExtent2D extent) {
		VkFramebufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		info.pNext = nullptr;

		info.renderPass = renderPass;
		info.attachmentCount = 1;
		info.width = extent.width;
		info.height = extent.height;
		info.layers = 1;

		return info;
	}

	VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags = 0) {
		VkFenceCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		info.pNext = nullptr;

		info.flags = flags;

		return info;
	}
	VkSemaphoreCreateInfo semaphore_create_info() {
		VkSemaphoreCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		info.pNext = nullptr;
		info.flags = 0;
		return info;
	}
	
	
	VkSubmitInfo submit_info(VkCommandBuffer* cmd) {
		VkSubmitInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		info.pNext = nullptr;

		info.waitSemaphoreCount = 0;
		info.pWaitSemaphores = nullptr;
		info.pWaitDstStageMask = nullptr;
		info.commandBufferCount = 1;
		info.pCommandBuffers = cmd;
		info.signalSemaphoreCount = 0;
		info.pSignalSemaphores = nullptr;

		return info;
	}
	VkPresentInfoKHR present_info() {
		VkPresentInfoKHR info = {};
		info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		info.pNext = nullptr;

		info.swapchainCount = 0;
		info.pSwapchains = nullptr;
		info.pWaitSemaphores = nullptr;
		info.waitSemaphoreCount = 0;
		info.pImageIndices = nullptr;

		return info;
	}
	VkRenderPassBeginInfo renderpass_begin_info(VkRenderPass renderPass,VkExtent2D windowExtent,VkFramebuffer framebuffer) {
		VkRenderPassBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		info.pNext = nullptr;

		info.renderPass = renderPass;
		info.renderArea.offset.x = 0;
		info.renderArea.offset.y = 0;
		info.renderArea.extent = windowExtent;
		info.clearValueCount = 1;
		info.pClearValues =nullptr;
		info.framebuffer = framebuffer;

		return info;
	}
	
	VkPipelineMultisampleStateCreateInfo multisampling_state_create_info()
	{
		VkPipelineMultisampleStateCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		info.pNext = nullptr;

		info.sampleShadingEnable = VK_FALSE;
		//multisampling defaulted to no multisampling (1 sample per pixel)
		info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		info.minSampleShading = 1.0f;
		info.pSampleMask = nullptr;
		info.alphaToCoverageEnable = VK_FALSE;
		info.alphaToOneEnable = VK_FALSE;
		return info;
	}

	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info(VkPolygonMode polygonMode)
	{
		VkPipelineRasterizationStateCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		info.pNext = nullptr;

		info.depthClampEnable = VK_FALSE;
		//rasterizer discard allows objects with holes, default to no
		info.rasterizerDiscardEnable = VK_FALSE;

		info.polygonMode = polygonMode;
		info.lineWidth = 1.0f;
		//backface cull enable, culling counter-clockwise faces
		info.cullMode = VK_CULL_MODE_BACK_BIT;
		info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		//no depth bias
		info.depthBiasEnable = VK_FALSE;
		info.depthBiasConstantFactor = 0.0f;
		info.depthBiasClamp = 0.0f; 
		info.depthBiasSlopeFactor = 0.0f; 

		return info;
	}

	VkPipelineColorBlendAttachmentState color_blend_attachment_state() {
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		return colorBlendAttachment;
	}

	VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info(VkPrimitiveTopology topology) {
		VkPipelineInputAssemblyStateCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		info.pNext = nullptr;

		info.topology = topology;
		info.primitiveRestartEnable = VK_FALSE;
		return info;
	}

	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info() {
		VkPipelineVertexInputStateCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		info.pNext = nullptr;

		//no vertex bindings or attributes
		info.vertexBindingDescriptionCount = 0;
		info.vertexAttributeDescriptionCount = 0;
		return info;
	}

	VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp) {
		VkPipelineDepthStencilStateCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		info.pNext = nullptr;

		info.depthTestEnable = bDepthTest ? VK_TRUE : VK_FALSE;
		info.depthWriteEnable = bDepthWrite ? VK_TRUE : VK_FALSE;
		info.depthCompareOp = bDepthTest ? compareOp : VK_COMPARE_OP_ALWAYS;
		info.depthBoundsTestEnable = VK_FALSE;
		info.minDepthBounds = 0.0f; // Optional
		info.maxDepthBounds = 1.0f; // Optional
		info.stencilTestEnable = VK_FALSE;
	
		return info;
	}
	VkImageCreateInfo image_create_info(VkFormat format , VkImageUsageFlags usageFlags ,VkExtent3D extent) {
		VkImageCreateInfo info = { };
		info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		info.pNext = nullptr;

		info.imageType = VK_IMAGE_TYPE_2D;		
		
		info.format = format;
		info.extent = extent;

		info.mipLevels = 1;
		info.arrayLayers = 1;
		info.samples = VK_SAMPLE_COUNT_1_BIT;
		info.tiling = VK_IMAGE_TILING_OPTIMAL;
		info.usage = usageFlags;

		return info;
	}
}


struct VertexInputDescription {
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {

	glm::vec3 position;
	glm::vec3 normal;

	static VertexInputDescription getVertexInputState() {
		VertexInputDescription description;

		VkVertexInputBindingDescription mainBinding = {};
		mainBinding.binding = 0;
		mainBinding.stride = sizeof(Vertex);
		mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		description.bindings.push_back(mainBinding);

		VkVertexInputAttributeDescription positionAttribute = {};
		positionAttribute.binding = 0;
		positionAttribute.location = 0;
		positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
		positionAttribute.offset = offsetof(Vertex, position);

		VkVertexInputAttributeDescription normalAttribute = {};
		normalAttribute.binding = 0;
		normalAttribute.location = 1;
		normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
		normalAttribute.offset = offsetof(Vertex, normal);

		description.attributes.push_back(positionAttribute);
		description.attributes.push_back(normalAttribute);

		return description;
	}
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

struct AllocatedImage {
	VkImage image;	
	VmaAllocation allocation;
};

namespace vkutil {

	Mesh load_mesh_from_obj(const std::string& filename) {
		std::vector<tinyobj::material_t> materials;
			
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;

		std::string warn;
		std::string err;
		bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(),
			nullptr);
		if (!warn.empty()) {
			std::cout << "WARN: " << warn << std::endl;
		}
		if (!err.empty()) {
			std::cerr << err << std::endl;
		}

		Mesh newMesh;
		// Loop over shapes
		for (size_t s = 0; s < shapes.size(); s++) {
			// Loop over faces(polygon)
			size_t index_offset = 0;
			for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				int fv = 3;

				// Loop over vertices in the face.
				for (size_t v = 0; v < fv; v++) {
					// access to vertex
					tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
					tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
					tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
					tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
					tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
					tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
					tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
					tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
					
					Vertex new_vert;
					new_vert.position.x = vx;
					new_vert.position.y = vy;
					new_vert.position.z = vz;

					new_vert.normal.x = nx;
					new_vert.normal.y = ny;
					new_vert.normal.z = nz;

					newMesh.indices.push_back(newMesh.vertices.size());
					newMesh.vertices.push_back(new_vert);					
				}
				index_offset += fv;				
			}
		}

		return newMesh;
	}

	VkRenderPass create_render_pass(VkDevice device, VkFormat image_format, VkFormat depth_format) {

		//we define an attachment description for our main color image
		//the attachment is loaded as "clear" when renderpass start
		//the attachment is stored when renderpass ends
		//the attachment layout starts as "undefined", and transitions to "Present" so its possible to display it
		//we dont care about stencil, and dont use multisampling

		VkAttachmentDescription color_attachment = {};
		color_attachment.format = image_format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depth_attachment = {};
		 // Depth attachment
		depth_attachment.flags = 0;
		depth_attachment.format = depth_format;
		depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		
		VkAttachmentDescription attachments[2] = { color_attachment,depth_attachment };

		VkAttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depth_attachment_ref = {};
		depth_attachment_ref.attachment = 1;
		depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		//we are going to create 1 subpass, which is the minimum you can do
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;
		subpass.pDepthStencilAttachment = &depth_attachment_ref;

		//1 dependency, which is from "outside" into the subpass. And we can read or write color
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


		VkRenderPassCreateInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 2;
		render_pass_info.pAttachments = attachments;//&color_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 1;
		render_pass_info.pDependencies = &dependency;

		VkRenderPass renderPass;
		VK_CHECK(vkCreateRenderPass(device, &render_pass_info, nullptr, &renderPass));
		return renderPass;
	}

	//loads a shader module from a spir-v file. Returns false if it errors
	bool load_shader_module(const std::string& filename, VkDevice device, VkShaderModule* outShaderModule) {

		//open the file. With cursor at the end
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			return false;
		}

		//find what the size of the file is by looking up the location of the cursor
		//because the cursor is at the end, it gives the size directly in bytes
		size_t fileSize = (size_t)file.tellg();

		//spirv expects the buffer to be on uint32, so make sure to reserve a int vector big enough for the entire file
		std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

		//put file cursor at beggining
		file.seekg(0);

		//load the entire file into the buffer
		file.read((char*)buffer.data(), fileSize);

		//now that the file is loaded into the buffer, we can close it
		file.close();

		//create a new shader module, using the buffer we loaded
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.pNext = nullptr;

		//codeSize has to be in bytes, so multply the ints in the buffer by size of int to know the real size of the buffer
		createInfo.codeSize = buffer.size() * sizeof(uint32_t); 
		createInfo.pCode = buffer.data();

		//check that the creation goes well.
		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			return false;
		}
		*outShaderModule = shaderModule;
		return true;
	}

	class PipelineBuilder {
	public:

		std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
		VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
		VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
		VkViewport _viewport;
		VkRect2D _scissor;
		VkPipelineRasterizationStateCreateInfo _rasterizer;
		VkPipelineColorBlendAttachmentState _colorBlendAttachment;
		VkPipelineMultisampleStateCreateInfo _multisampling;
		VkPipelineLayout _pipelineLayout;
		VkPipelineDepthStencilStateCreateInfo _depthStencil;

		VkPipeline build_pipeline(VkDevice device, VkRenderPass pass) {
			
			//make viewport state from our stored viewport and scissor.
			//at the moment we wont support multiple viewports or scissors
			VkPipelineViewportStateCreateInfo viewportState = {};
			viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportState.pNext = nullptr;

			viewportState.viewportCount = 1;
			viewportState.pViewports = &_viewport;
			viewportState.scissorCount = 1;
			viewportState.pScissors = &_scissor;

			//setup dummy color blending. We arent using transparent objects yet
			//the blending is just "no blend", but we do write to the color attachment
			VkPipelineColorBlendStateCreateInfo colorBlending = {};
			colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			colorBlending.pNext = nullptr;

			colorBlending.logicOpEnable = VK_FALSE;
			colorBlending.logicOp = VK_LOGIC_OP_COPY;
			colorBlending.attachmentCount = 1;
			colorBlending.pAttachments = &_colorBlendAttachment;
			colorBlending.blendConstants[0] = 0.0f;
			colorBlending.blendConstants[1] = 0.0f;
			colorBlending.blendConstants[2] = 0.0f;
			colorBlending.blendConstants[3] = 0.0f;

			//build the actual pipeline
			//we now use all of the info structs we have been writing into into this one to create the pipeline
			VkGraphicsPipelineCreateInfo pipelineInfo = {};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.pNext = nullptr;

			pipelineInfo.stageCount = _shaderStages.size();
			pipelineInfo.pStages = _shaderStages.data();
			pipelineInfo.pVertexInputState = &_vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &_inputAssembly;
			pipelineInfo.pViewportState = &viewportState;
			pipelineInfo.pRasterizationState = &_rasterizer;
			pipelineInfo.pMultisampleState = &_multisampling;
			pipelineInfo.pColorBlendState = &colorBlending;
			pipelineInfo.pDepthStencilState = &_depthStencil;
			pipelineInfo.layout = _pipelineLayout;
			pipelineInfo.renderPass = pass;
			pipelineInfo.subpass = 0;
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

			//its easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
			VkPipeline newPipeline;
			if (vkCreateGraphicsPipelines(
				device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
				std::cout << "failed to create pipline\n";
				return VK_NULL_HANDLE; // failed to create graphics pipeline
			}

			return newPipeline;
		}
	};
	bool create_mesh_pipeline(VkDevice device, VkExtent2D swapchainExtent, VkRenderPass renderPass, VkPipelineLayout layout, const std::string& vertex_shader, const std::string& frag_shader, VkPipeline* outPipeline) {

		PipelineBuilder pipelineBuilder;

		VkShaderModule vert_module;
		VkShaderModule frag_module;

		//load the fragment and vertex shaders for the triangle
		//if any of the 2 give error we abort
		if (!load_shader_module(vertex_shader, device, &vert_module) ||
			!load_shader_module(frag_shader, device, &frag_module)) {
			std::cout << "failed to create shader module\n";
			return false;
		}

		//build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
		VkPipelineShaderStageCreateInfo vert_stage_info = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vert_module);

		VkPipelineShaderStageCreateInfo frag_stage_info = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, frag_module);

		pipelineBuilder._shaderStages.push_back(vert_stage_info);
		pipelineBuilder._shaderStages.push_back(frag_stage_info);


		//vertex input controls how to read vertices from vertex buffers.
		pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

		VertexInputDescription vertexDescription = Vertex::getVertexInputState();

		pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
		pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

		pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
		pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

		//input assembly is the configuration for drawing triangle lists, strips, or individual points.
		//we are just going to draw triangle list
		pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		//build viewport and scissor from the swapchain extents
		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapchainExtent.width;
		viewport.height = (float)swapchainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapchainExtent;

		pipelineBuilder._viewport = viewport;

		pipelineBuilder._scissor = scissor;

		//configure the rasterizer to draw filled triangles with normal culling
		pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

		//we dont use multisampling, so just run the default one
		pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

		//a single blend attachment with no blending and writing to RGBA
		pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();

		pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

		pipelineBuilder._pipelineLayout = layout;

		//finally build the pipeline
		VkPipeline newPipeline = pipelineBuilder.build_pipeline(device, renderPass);


		//clean up the loaded shader modules, once the pipeline is built we no longer need it
		vkDestroyShaderModule(device, frag_module, nullptr);
		vkDestroyShaderModule(device, vert_module, nullptr);

		//check that the pipeline was build correctly
		if (newPipeline != VK_NULL_HANDLE) {
			*outPipeline = newPipeline;
			return true;
		}
		else {
			return false;
		}
	}

	bool create_triangle_pipeline(VkDevice device, VkExtent2D swapchainExtent, VkRenderPass renderPass,VkPipelineLayout layout,const std::string& vertex_shader, const std::string& frag_shader , VkPipeline* outPipeline) {
		
		PipelineBuilder pipelineBuilder;

		VkShaderModule vert_module;
		VkShaderModule frag_module;

		//load the fragment and vertex shaders for the triangle
		//if any of the 2 give error we abort
		if (!load_shader_module(vertex_shader,device,&vert_module) ||
			!load_shader_module(frag_shader, device, &frag_module)) {
			std::cout << "failed to create shader module\n";
			return false;
		}

		//build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
		VkPipelineShaderStageCreateInfo vert_stage_info = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vert_module);	

		VkPipelineShaderStageCreateInfo frag_stage_info = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, frag_module);		

		pipelineBuilder._shaderStages.push_back(vert_stage_info);
		pipelineBuilder._shaderStages.push_back(frag_stage_info);

		//vertex input controls how to read vertices from vertex buffers. We arent using it yet
		pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
		
		//input assembly is the configuration for drawing triangle lists, strips, or individual points.
		//we are just going to draw triangle list
		pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		//build viewport and scissor from the swapchain extents
		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapchainExtent.width;
		viewport.height = (float)swapchainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapchainExtent;

		pipelineBuilder._viewport = viewport;

		pipelineBuilder._scissor = scissor;

		//configure the rasterizer to draw filled triangles with normal culling
		pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

		//we dont use multisampling, so just run the default one
		pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

		//a single blend attachment with no blending and writing to RGBA
		pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();		

		pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(false, false, VK_COMPARE_OP_ALWAYS);

		pipelineBuilder._pipelineLayout = layout;

		//finally build the pipeline
		VkPipeline newPipeline = pipelineBuilder.build_pipeline(device, renderPass);


		//clean up the loaded shader modules, once the pipeline is built we no longer need it
		vkDestroyShaderModule(device, frag_module, nullptr);
		vkDestroyShaderModule(device, vert_module, nullptr);

		//check that the pipeline was build correctly
		if (newPipeline != VK_NULL_HANDLE) {
			*outPipeline = newPipeline;
			return true;
		}	
		else {
			return false;
		}
	}

	VkDescriptorPool create_imgui_descriptor_pool(VkDevice device) {
		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
		};
		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
		pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;

		VkDescriptorPool pool;
		VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool));

		return pool;
	}
	
}

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	
	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}
	}
};
class VulkanEngine {
public:

	
	VkInstance _instance;
	VkDevice _device;
	VkDebugUtilsMessengerEXT _debugMessenger;	
	
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;	

	VkQueue _graphicsQueue;
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkSurfaceKHR _surface;
	VkRenderPass _renderPass;
	VkExtent2D _windowExtent;
	VkSwapchainKHR _swapchain;
	std::vector<VkFramebuffer> _framebuffers;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	AllocatedImage _depthImage;
	VkImageView _depthImageView;

	VkPipeline _trianglePipeline;
	VkPipeline _funkTrianglePipeline;
	VkPipeline _meshPipeline;

	VkPipelineLayout _trianglePipelineLayout;
	VkPipelineLayout _meshPipelineLayout;
	Mesh monkey;
	VkBuffer _monkey;

	uint64_t _frameNumber;
	bool _isInitialized = false;

	bool _drawFunky = false;

	glm::vec3 camPos;

	DeletionQueue _mainDeletionQueue;

	void init();
	void cleanup();
	void draw();
};

SDL_Window* gWindow;
void VulkanEngine::init()
{
	_frameNumber = 0;

	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
	
	_windowExtent.height = 900;
	_windowExtent.width = 1700;

	gWindow = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_windowExtent.width,
		_windowExtent.height,
		window_flags
	);
	assert(gWindow != nullptr);

	vkb::InstanceBuilder builder;

	//make the vulkan instance, with basic debug features
	auto inst_ret = builder.set_app_name("Example Vulkan Application")
		.request_validation_layers(bUseValidationLayers)
		.use_default_debug_messenger()
		.build();
	
	vkb::Instance vkb_inst = inst_ret.value();

	//grab the instance and debug messenger
	_instance = vkb_inst.instance;
	_debugMessenger = vkb_inst.debug_messenger;
	
	

	//request a Vulkan surface from SDL, this is the actual drawable window output
	
	if (!SDL_Vulkan_CreateSurface(gWindow, _instance, &_surface)) {
		throw std::runtime_error("Failed to create surface");
		// failed to create a surface!
	}

	_mainDeletionQueue.push_function([=]() {
		//vkDestroySurfaceKHR(_instance, _surface, nullptr);
	});

	//use vkbootstrap to select a gpu. 
	//We want a gpu that can write to the SDL surface and supports vulkan 1.2
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	auto phys_ret = selector
		.set_minimum_version(1, 1)
		.add_required_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)
		.set_surface(_surface)
		.select();

	//create the final vulkan device

	vkb::DeviceBuilder deviceBuilder{ phys_ret.value() };	
	
	vkb::Device vkbDevice = deviceBuilder.build().value();
	VkPhysicalDevice physDevice = phys_ret.value().physical_device;

	// Get the VkDevice handle used in the rest of a vulkan application
	_device = vkbDevice.device;

	//add the destruction of device and instance to the queue
	_mainDeletionQueue.push_function([=]() {
		vkDestroyDevice(_device, nullptr);
		vkDestroyInstance(_instance, nullptr);
	});

	//now we begin to create the swapchain. We are going to use the lib so it configures everything for us
	//we want a swapchain with the same size as the SDL window surface, and with default optimal formats

	vkb::SwapchainBuilder swapchainBuilder{ vkbDevice };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		.build()
		.value();

	VmaAllocator allocator;

	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = phys_ret.value().physical_device;
	allocatorInfo.device = _device;	
	allocatorInfo.instance = _instance;
	vmaCreateAllocator(&allocatorInfo, &allocator);

	//add the destruction of allocator
	_mainDeletionQueue.push_function([=]() {
		vmaDestroyAllocator(allocator);
	});

	//store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews= vkbSwapchain.get_image_views().value();

	//add the destruction of swapchain
	_mainDeletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
	});

	//find a depth-buffer format to use
	VkFormat formats[] = {
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM
	};

	VkFormat selectedDepthFormat = VK_FORMAT_UNDEFINED;
	for (int i = 0; i < sizeof(formats) / sizeof(VkFormat); i++) {
		VkFormatProperties cfg;
		vkGetPhysicalDeviceFormatProperties(physDevice, formats[i], &cfg);
		if (cfg.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			selectedDepthFormat = formats[i];
			break;
		}
	}

	VkExtent3D depthImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};

	//the depth image will be a image with the format we selected and Depth Attachment usage flag
	VkImageCreateInfo dimg_info = vkinit::image_create_info(selectedDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);	

	//for the depth image, we want to allocate it from gpu local memory
	VmaAllocationCreateInfo dimg_allocinfo = {};	
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	
	//allocate and create the image
	vmaCreateImage(allocator, &dimg_info, &dimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

	//build a image-view for the depth image to use for rendering
	VkImageViewCreateInfo dview_info = {};
	dview_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	dview_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	dview_info.image = _depthImage.image;
	dview_info.format = selectedDepthFormat;
	dview_info.subresourceRange.baseMipLevel = 0;
	dview_info.subresourceRange.levelCount = 1;
	dview_info.subresourceRange.baseArrayLayer = 0;
	dview_info.subresourceRange.layerCount = 1;
	dview_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

	
	
	VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

	_mainDeletionQueue.push_function([=]() {
		vmaDestroyImage(allocator, _depthImage.image, _depthImage.allocation);
		vkDestroyImageView(_device, _depthImageView, nullptr);
	});

	//build the default render-pass we need to do rendering
	_renderPass = vkutil::create_render_pass(_device, vkbSwapchain.image_format,selectedDepthFormat);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _renderPass, nullptr);
	});

	//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_renderPass,_windowExtent);

	const uint32_t swapchain_imagecount = vkbSwapchain.image_count;
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	for (int i = 0; i < swapchain_imagecount; i++) {
		
		std::array<VkImageView, 2> attachments;
		attachments[0] = _swapchainImageViews[i];
		attachments[1] = _depthImageView;
		fb_info.pAttachments = attachments.data();
		fb_info.attachmentCount = 2;
		VK_CHECK( vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));		
	}

	//add the destruction of framebuffers
	_mainDeletionQueue.push_function([=]() {	

		//destroy swapchain resources
		for (int i = 0; i < _framebuffers.size(); i++) {
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);

			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
		}
	});

	// use vkbootstrap to get a Graphics queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();


	uint32_t graphics_queue_family = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	//create a command pool for commands submitted to the graphics queue.
	//we also want the pool to allow for resetting of individual command buffers
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(graphics_queue_family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

	//allocate the default command buffer that we will use for rendering
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_commandPool, 1);

	VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));

	//add the destruction of command pool. Queue and buffers dont have to get deleted
	_mainDeletionQueue.push_function([=]() {	

		vkDestroyCommandPool(_device, _commandPool, nullptr);
	});

	//create syncronization structures
	//one fence to control when the gpu has finished rendering the frame,
	//and 2 semaphores to syncronize rendering with swapchain
	//we want the fence to start signalled so we can wait on it on the first frame
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info();

	VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));
	
	//add the destruction of sync primitives
	_mainDeletionQueue.push_function([=]() {

		vkDestroyFence(_device, _renderFence, nullptr);
		vkDestroySemaphore(_device, _renderSemaphore, nullptr);
		vkDestroySemaphore(_device, _presentSemaphore, nullptr);
	});

	//build the pipeline layout that controls the inputs/outputs of the shader	
	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();

	//setup push constants
	VkPushConstantRange push_constant;
	//offset 0
	push_constant.offset = 0;
	//size of 4 floats
	push_constant.size = sizeof(float) * 4;
	//for the fragment shader
	push_constant.stageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	pipeline_layout_info.pPushConstantRanges = &push_constant;
	pipeline_layout_info.pushConstantRangeCount = 1;

	VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));
	
	vkutil::create_triangle_pipeline(_device, _windowExtent, _renderPass, _trianglePipelineLayout,
		"../../shaders/triangle.vert.spv",
		"../../shaders/triangle.frag.spv",
		&_trianglePipeline);

	vkutil::create_triangle_pipeline(_device, _windowExtent, _renderPass, _trianglePipelineLayout,
		"../../shaders/triangle.vert.spv",
		"../../shaders/funky_triangle.frag.spv",
		&_funkTrianglePipeline);

	push_constant.size = sizeof(glm::mat4);
	push_constant.stageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;

	VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

	vkutil::create_mesh_pipeline(_device, _windowExtent, _renderPass, _meshPipelineLayout,
		"../../shaders/mesh.vert.spv",
		"../../shaders/mesh.frag.spv",
		&_meshPipeline);

	//add the destruction of pipelines and layouts
	_mainDeletionQueue.push_function([=]() {
		
		vkDestroyPipeline(_device, _trianglePipeline, nullptr);
		vkDestroyPipeline(_device, _funkTrianglePipeline, nullptr);
		vkDestroyPipeline(_device, _meshPipeline, nullptr);

		vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
		vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
	});
	
   monkey = vkutil::load_mesh_from_obj("../../assets/monkey_smooth.obj");

   //allocate vertex buffer
   VkBufferCreateInfo bufferInfo = {};
   bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
   bufferInfo.size = monkey.vertices.size() * sizeof(Vertex);
   bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
   bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

   VkBufferCreateInfo vkbinfo = bufferInfo;

   VmaAllocationCreateInfo vmaallocInfo = {};
   vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

   VmaAllocation allocation;   
   VK_CHECK(vmaCreateBuffer(allocator, &vkbinfo, &vmaallocInfo, &_monkey, &allocation, nullptr));

   //add the destruction of mesh buffer
   _mainDeletionQueue.push_function([=]() {
	
	   vmaDestroyBuffer(allocator, _monkey, allocation);
	});

   //copy vertex data
   void* data;
   vmaMapMemory(allocator, allocation, &data);

   memcpy(data, monkey.vertices.data(), monkey.vertices.size() * sizeof(Vertex));

   vmaUnmapMemory(allocator, allocation);

   VkDescriptorPool imguiPool = vkutil::create_imgui_descriptor_pool(_device);

   ImGui::CreateContext();

   ImGui_ImplSDL2_InitForVulkan(gWindow);

   ImGui_ImplVulkan_InitInfo init_info = {};
   init_info.Instance = _instance;
   init_info.PhysicalDevice = physDevice;
   init_info.Device = _device;   
   init_info.Queue = _graphicsQueue;  
   init_info.DescriptorPool = imguiPool;   
   init_info.MinImageCount = 3;
   init_info.ImageCount = 3;

   ImGui_ImplVulkan_Init(&init_info, _renderPass);

   //add the destruction of mesh buffer
   _mainDeletionQueue.push_function([=]() {

	   vkDestroyDescriptorPool(_device, imguiPool, nullptr);
	   ImGui_ImplVulkan_Shutdown();
   });

   //naming it cmd for shorter writing
   VkCommandBuffer cmd = _mainCommandBuffer;

   //begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
   VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

   VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));   
   
   ImGui_ImplVulkan_CreateFontsTexture(cmd);

   VK_CHECK(vkEndCommandBuffer(cmd));  

   VkSubmitInfo submit = vkinit::submit_info(&cmd);
   VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

   submit.pWaitDstStageMask = &waitStage;

   //submit command buffer to the queue and execute it.
   // _renderFence will now block until the graphic commands finish execution
   VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

   //wait until the gpu has finished uploading
   VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
  
   ImGui_ImplVulkan_DestroyFontUploadObjects();
	//everything went fine
	_isInitialized = true;
}
void VulkanEngine::draw() {	
	
	//wait until the gpu has finished rendering the last frame. Timeout of 1 second
	VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &_renderFence));

	//now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
	VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));
	
	//request image from the swapchain
	uint32_t swapchainImageIndex;
	VK_CHECK( vkAcquireNextImageKHR(_device, _swapchain, 0, _presentSemaphore, nullptr , &swapchainImageIndex));	
	
	//naming it cmd for shorter writing
	VkCommandBuffer cmd = _mainCommandBuffer;

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	//make a clear-color from frame number. This will flash with a 120 frame period.
	std::array<VkClearValue, 2> clearValues;
	float flash = abs(sin(_frameNumber / 120.f));
	clearValues[0].color = { { 0.0f, 0.0f, flash, 1.0f } };

	clearValues[1].depthStencil.depth = 1.f;

	//start the main renderpass. 
	//We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass,_windowExtent,_framebuffers[swapchainImageIndex]);
	
	//connect clear values
	rpInfo.clearValueCount = 2;
	rpInfo.pClearValues = clearValues.data();	

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	//once we start adding rendering commands, they will go here

	if (_drawFunky)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _funkTrianglePipeline);
		std::array<float, 4> pushConstantData{ 0,0,0,0 };
		pushConstantData[0] = _frameNumber / 120.f;

		vkCmdPushConstants(cmd, _trianglePipelineLayout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, sizeof(float) * 4, pushConstantData.data());

		vkCmdDraw(cmd, 3, 1, 0, 0);
	}
	else {

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

		//bind the mesh vertex buffer with offset 0
		VkDeviceSize offset = 0;		
		vkCmdBindVertexBuffers(cmd, 0, 1,&_monkey,&offset);
		
		//make a model view matrix for rendering the object
		//camera view
		glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);		
		//camera projection
		glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
		projection[1][1] *= -1;
		//model rotation
		glm::mat4 model = glm::rotate(glm::mat4{ 0.1f }, glm::radians(_frameNumber * 0.4f), glm::vec3(0, 1, 0));

		glm::mat4 mesh_matrix = projection* view * model;

		//upload the mesh to the gpu via pushconstants
		vkCmdPushConstants(cmd, _meshPipelineLayout, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, sizeof(glm::mat4), &mesh_matrix);

		//we can now draw
		vkCmdDraw(cmd, monkey.indices.size(), 1, 0, 0);
	}

	
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
	//finalize the render pass
	vkCmdEndRenderPass(cmd);
	//finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(cmd));

	//prepare the submission to the queue. 
	//we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
	//we will signal the _renderSemaphore, to signal that rendering has finished

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	
	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &_presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &_renderSemaphore;

	//submit command buffer to the queue and execute it.
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

	//prepare present
	// this will put the image we just rendered to into the visible window.
	// we want to wait on the _renderSemaphore for that, 
	// as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo = vkinit::present_info();

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;
	
	presentInfo.pWaitSemaphores = &_renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK (vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	//increase the number of frames drawn
	_frameNumber++;
}
void VulkanEngine::cleanup()
{	
	//make sure the gpu has stopped doing its things
	vkWaitForFences(_device, 1, &_renderFence, true, 999999999);


	
	_mainDeletionQueue.flush();
	
	SDL_DestroyWindow(gWindow);
}

int main(int argc, char* argv[])
{
	VulkanEngine engine;
	engine.init();
	engine.camPos = glm::vec3(0);
	engine.camPos.z = -3;

	SDL_Event e;
	bool bQuit = false;
	bool bStopRender = false;
	//main loop
	while (!bQuit)
	{		
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL2_NewFrame(gWindow);

		ImGui::NewFrame();

		static bool bShowDemo = true;

		ImGui::ShowDemoWindow(&bShowDemo);	

		ImGui::Render();

		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			ImGui_ImplSDL2_ProcessEvent(&e);


			//close the window when user alt-f4s or clicks the X button
			if (e.type == SDL_WINDOWEVENT)
			{
				switch (e.window.event) {
				case SDL_WINDOWEVENT_MINIMIZED:
					bStopRender = true;
					break;
					
				case SDL_WINDOWEVENT_RESTORED:
					bStopRender = false;
					break;
				}
			}
			if (e.type == SDL_QUIT) bQuit = true;
			else if (e.type == SDL_KEYDOWN)
			{
				switch (e.key.keysym.sym)
				{
				case SDLK_SPACE:
					engine._drawFunky = !engine._drawFunky;
					break;
				case SDLK_a:
					engine.camPos.x += 0.1;
					break;
				case SDLK_d:
					engine.camPos.x -= 0.1;
					break;

				case SDLK_w:
					engine.camPos.z += 0.1;
					break;
				case SDLK_s:
					engine.camPos.z -= 0.1;
					break;
				}				
			}
		}

		if (!bStopRender) {

			engine.draw();
		}
	}

	if (engine._isInitialized) {

		//make sure to release the resources of the engine properly if it was initialized well		
		engine.cleanup();
	}	

	return 0;
}
