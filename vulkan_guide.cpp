#include "volk.h"
#define VK_NO_PROTOTYPES

#include "VkBootstrap.h"

#include "vulkan_guide.h"
#include <vector>
#include <array>
#include <fstream>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <glm/glm.hpp>

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
		info.frontFace = VK_FRONT_FACE_CLOCKWISE;
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



	
}



namespace vkutil {

	VkRenderPass create_render_pass(VkDevice device, VkFormat image_format) {

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

		VkAttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		//we are going to create 1 subpass, which is the minimum you can do
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

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
		render_pass_info.attachmentCount = 1;
		render_pass_info.pAttachments = &color_attachment;
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
}

struct VertexInputDescription {
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags=0;
};

struct Vertex {

	std::array<float, 3> position;
	std::array<float, 3> color;

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

		VkVertexInputAttributeDescription colorAttribute = {};
		positionAttribute.binding = 0;
		positionAttribute.location = 1;
		positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
		positionAttribute.offset = offsetof(Vertex, color);

		description.attributes.push_back(positionAttribute);
		description.attributes.push_back(colorAttribute);
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
	
	VkPipeline _trianglePipeline;
	VkPipeline _funkTrianglePipeline;

	VkPipelineLayout _trianglePipelineLayout;

	

	uint64_t _frameNumber;
	bool _isInitialized = false;

	bool _drawFunky = false;

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
	
	//Volk needs a pre-initialization before creating the vulkan instance, so that it can load the functions needed to init vulkan
	volkInitialize();

	vkb::InstanceBuilder builder;

	//make the vulkan instance, with basic debug features
	auto inst_ret = builder.set_app_name("Example Vulkan Application")
		.request_validation_layers(bUseValidationLayers)
		.use_default_debug_messenger()
		.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT)
		//.add_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT)
		//.add_debug_messenger_type(VK_DEBUG_UTILS_MESSAGE_TYPE_FLAG_BITS_MAX_ENUM_EXT)
		.build();
	
	vkb::Instance vkb_inst = inst_ret.value();

	//grab the instance and debug messenger
	_instance = vkb_inst.instance;
	_debugMessenger = vkb_inst.debug_messenger;
	
	//now that instance is loaded, use volk to load all the vulkan functions and extensions
	volkLoadInstance(_instance);

	

	//request a Vulkan surface from SDL, this is the actual drawable window output
	
	if (!SDL_Vulkan_CreateSurface(gWindow, _instance, &_surface)) {
		throw std::runtime_error("Failed to create surface");
		// failed to create a surface!
	}

	//use vkbootstrap to select a gpu. 
	//We want a gpu that can write to the SDL surface and supports vulkan 1.2
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	auto phys_ret = selector
		.set_minimum_version(1, 2)
		.add_required_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)
		.set_surface(_surface)
		.select();

	//create the final vulkan device

	vkb::DeviceBuilder deviceBuilder{ phys_ret.value() };	
	
	vkb::Device vkbDevice = deviceBuilder.build().value();

	// Get the VkDevice handle used in the rest of a vulkan application
	_device = vkbDevice.device;

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

	
	//store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews= vkbSwapchain.get_image_views().value();

	//build the default render-pass we need to do rendering
	_renderPass = vkutil::create_render_pass(_device, vkbSwapchain.image_format);


	//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_renderPass,_windowExtent);

	const uint32_t swapchain_imagecount = vkbSwapchain.image_count;
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	for (int i = 0; i < swapchain_imagecount; i++) {
		
		fb_info.pAttachments = &_swapchainImageViews[i];
		VK_CHECK( vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));		
	}

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

	//create syncronization structures
	//one fence to control when the gpu has finished rendering the frame,
	//and 2 semaphores to syncronize rendering with swapchain
	//we want the fence to start signalled so we can wait on it on the first frame
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);

	VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));
	

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
		"C:/Programming/vulkan-guide/shaders/triangle.vert.spv",
		"C:/Programming/vulkan-guide/shaders/triangle.frag.spv",
		&_trianglePipeline);

	vkutil::create_triangle_pipeline(_device, _windowExtent, _renderPass, _trianglePipelineLayout,
		"C:/Programming/vulkan-guide/shaders/triangle.vert.spv",
		"C:/Programming/vulkan-guide/shaders/funky_triangle.frag.spv",
		&_funkTrianglePipeline);

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
	VkClearValue clearValue;
	float flash = abs(sin(_frameNumber / 120.f));
	clearValue.color = { { 0.0f, 0.0f, flash, 1.0f } };	

	//start the main renderpass. 
	//We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass,_windowExtent,_framebuffers[swapchainImageIndex]);
	
	//connect clear values
	rpInfo.clearValueCount = 1;
	rpInfo.pClearValues = &clearValue;	

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	//once we start adding rendering commands, they will go here

	if (_drawFunky)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _funkTrianglePipeline);
	}
	else {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _trianglePipeline);
	}

	std::array<float, 4> pushConstantData{0,0,0,0};
	pushConstantData[0] = _frameNumber / 120.f;
	
	vkCmdPushConstants(cmd, _trianglePipelineLayout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, sizeof(float) * 4, pushConstantData.data());

	vkCmdDraw(cmd, 3, 1, 0, 0);

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

	vkDestroyCommandPool(_device, _commandPool, nullptr);

	//destroy the pipeline and its layout
	vkDestroyPipeline(_device, _trianglePipeline, nullptr);
	vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);

	//destroy sync objects
	vkDestroyFence(_device, _renderFence, nullptr);
	vkDestroySemaphore(_device, _renderSemaphore, nullptr);
	vkDestroySemaphore(_device, _presentSemaphore, nullptr);
	
	vkDestroySwapchainKHR(_device, _swapchain, nullptr);

	vkDestroyRenderPass(_device, _renderPass, nullptr);
	
	//destroy swapchain resources
	for (int i = 0; i < _framebuffers.size(); i++) {
		vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);

		vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
	}

	vkDestroySurfaceKHR(_instance, _surface, nullptr);

	//destroy debug utils
	vkDestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);

	vkDestroyDevice(_device, nullptr);
	vkDestroyInstance(_instance,nullptr);	
	
	SDL_DestroyWindow(gWindow);
}

int main(int argc, char* argv[])
{
	VulkanEngine engine;
	engine.init();

	SDL_Event e;
	bool bQuit = false;
	//main loop
	while (!bQuit)
	{		
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			//close the window when user alt-f4s or clicks the X button
			if (e.type == SDL_QUIT) bQuit = true;
			else if (e.type == SDL_KEYDOWN)
			{
				if (e.key.keysym.sym == SDLK_SPACE)
				{
					engine._drawFunky = !engine._drawFunky;
				}
			}
		}

		engine.draw();	
	}

	if (engine._isInitialized) {

		//make sure to release the resources of the engine properly if it was initialized well		
		engine.cleanup();
	}	

	return 0;
}
