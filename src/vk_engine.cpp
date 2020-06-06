#include "VkBootstrap.h"

#include "vk_engine.h"
#include <vector>
#include <array>
#include <fstream>
#include <functional>
#include <deque>
#include <iostream>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>



#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>

#include <vk_types.h>
#include <vk_mesh.h>
#include <vk_initializers.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

//set to false to disable validation layers
const bool bUseValidationLayers = true;

struct SDL_Window* gWindow = nullptr;

namespace vkutil{
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
		pool_info.maxSets = 1000;
		pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;

		VkDescriptorPool pool;
		VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool));

		return pool;
	}	
}


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
	_physicalDevice = phys_ret.value().physical_device;

	// Get the VkDevice handle used in the rest of a vulkan application
	_device = vkbDevice.device;

	//add the destruction of device and instance to the queue
	_mainDeletionQueue.push_function([=]() {
		vkDestroyDevice(_device, nullptr);
		vkDestroyInstance(_instance, nullptr);
	});

	//initialize the memory allocator
	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = phys_ret.value().physical_device;
	allocatorInfo.device = _device;
	allocatorInfo.instance = _instance;
	vmaCreateAllocator(&allocatorInfo, &_allocator);


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

	//add the destruction of allocator
	_mainDeletionQueue.push_function([=]() {
		vmaDestroyAllocator(_allocator);
	});

	//store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews= vkbSwapchain.get_image_views().value();

	//add the destruction of swapchain
	_mainDeletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
	});


	//create depth image
	VkFormat selectedDepthFormat = select_depth_format();

	init_depth_image(selectedDepthFormat);


	//build the default render-pass we need to do rendering
	_renderPass = vkutil::create_render_pass(_device, vkbSwapchain.image_format,selectedDepthFormat);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _renderPass, nullptr);
	}); 
	
	init_framebuffers(vkbSwapchain.image_count);

	// use vkbootstrap to get a Graphics queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();

	//initialize the commands with a queue index for that graphics queue
	uint32_t graphics_queue_family = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	init_commands(graphics_queue_family);

	init_syncronization_structures();

	init_pipelines();	

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

   vkutil::load_mesh_from_obj("../../assets/monkey_smooth.obj",vertices,indices);
   
   upload_mesh(vertices, indices,_monkeyMesh);  

   init_imgui();

	//everything went fine
	_isInitialized = true;
}

void VulkanEngine::init_commands(uint32_t graphics_queue_family)
{
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
}

void VulkanEngine::init_syncronization_structures()
{
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
}

VkFormat VulkanEngine::select_depth_format()
{
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
		vkGetPhysicalDeviceFormatProperties(_physicalDevice, formats[i], &cfg);
		if (cfg.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			return formats[i];			
		}
	}
	return VK_FORMAT_UNDEFINED;
}

void VulkanEngine::init_imgui()
{
	VkDescriptorPool imguiPool = vkutil::create_imgui_descriptor_pool(_device);

	ImGui::CreateContext();

	ImGui_ImplSDL2_InitForVulkan(gWindow);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = _instance;
	init_info.PhysicalDevice = _physicalDevice;
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
}

void VulkanEngine::init_framebuffers(int swapchain_imagecount)
{
	//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_renderPass, _windowExtent);

	
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	for (int i = 0; i < swapchain_imagecount; i++) {

		std::array<VkImageView, 2> attachments;
		attachments[0] = _swapchainImageViews[i];
		attachments[1] = _depthImageView;
		fb_info.pAttachments = attachments.data();
		fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));
	}

	//add the destruction of framebuffers
	_mainDeletionQueue.push_function([=]() {

		//destroy swapchain resources
		for (int i = 0; i < _framebuffers.size(); i++) {
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);

			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
		}
	});
}

void VulkanEngine::init_pipelines()
{
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

	//add the destruction of pipelines and their layouts
	_mainDeletionQueue.push_function([=]() {

		vkDestroyPipeline(_device, _trianglePipeline, nullptr);
		vkDestroyPipeline(_device, _funkTrianglePipeline, nullptr);
		vkDestroyPipeline(_device, _meshPipeline, nullptr);

		vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
		vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
	});
}

void VulkanEngine::init_depth_image(VkFormat selectedDepthFormat)
{
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
	vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

	//build a image-view for the depth image to use for rendering
	VkImageViewCreateInfo dview_info = {};
	dview_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	dview_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	dview_info.image = _depthImage._image;
	dview_info.format = selectedDepthFormat;
	dview_info.subresourceRange.baseMipLevel = 0;
	dview_info.subresourceRange.levelCount = 1;
	dview_info.subresourceRange.baseArrayLayer = 0;
	dview_info.subresourceRange.layerCount = 1;
	dview_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

	VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

	_mainDeletionQueue.push_function([=]() {
		vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
		vkDestroyImageView(_device, _depthImageView, nullptr);
	});
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

		_monkeyMesh.bind_vertex_buffer(cmd);
		
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
		vkCmdDraw(cmd, _monkeyMesh._indices.size(), 1, 0, 0);
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

bool VulkanEngine::upload_mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Mesh& outMesh)
{
	outMesh._vertices = vertices;
	outMesh._indices = indices;

	//allocate vertex buffer
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = _monkeyMesh._vertices.size() * sizeof(Vertex);
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBufferCreateInfo vkbinfo = bufferInfo;

	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	VK_CHECK(vmaCreateBuffer(_allocator, &vkbinfo, &vmaallocInfo, &outMesh._vertexBuffer._buffer, &outMesh._vertexBuffer._allocation, nullptr));

	//add the destruction of mesh buffer
	_mainDeletionQueue.push_function([=]() {

		vmaDestroyBuffer(_allocator , outMesh._vertexBuffer._buffer, outMesh._vertexBuffer._allocation);
	});


	//copy vertex data
	void* data;
	vmaMapMemory(_allocator, outMesh._vertexBuffer._allocation, &data);

	memcpy(data, _monkeyMesh._vertices.data(), _monkeyMesh._vertices.size() * sizeof(Vertex));

	vmaUnmapMemory(_allocator, outMesh._vertexBuffer._allocation);

	return true;
}

void VulkanEngine::cleanup()
{	
	//make sure the gpu has stopped doing its things
	vkWaitForFences(_device, 1, &_renderFence, true, 999999999);
	
	_mainDeletionQueue.flush();
	
	SDL_DestroyWindow(gWindow);
}

