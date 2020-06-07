// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_mesh.h>
#include <vector>
#include <vk_types.h>
#include <deque>
#include <functional>



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

		deletors.clear();
	}
};

extern struct SDL_Window* gWindow;

struct alignas(256) WorldParameters{
	glm::mat4 cameraMatrix; //viewproj
	glm::vec4 ambient_color;
};

struct alignas(256) ObjectUniforms {
	glm::mat4 modelMatrix;
	glm::vec4 shine_color;
};

class VulkanEngine {
public:

	VmaAllocator _allocator;
	VkInstance _instance;
	VkDevice _device;
	VkPhysicalDevice _physicalDevice;
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

	VkDescriptorSetLayout _singleUniformSetLayout;
	VkDescriptorSetLayout _singleUniformDynamicSetLayout;

	VkDescriptorPool _frameDescriptorPool;

	Mesh _monkeyMesh;

	//holds uniform data for world parameters
	AllocatedBuffer _worldParameterBuffer;

	//holds uniform data for objects
	AllocatedBuffer _objectDataBuffer;

	const uint32_t max_monkeys = 20;
	int _numMonkeys = 5;

	std::vector<ObjectUniforms> _meshUniforms;

	uint64_t _frameNumber;
	bool _isInitialized = false;

	bool _drawFunky = false;

	glm::vec3 camPos;

	DeletionQueue _mainDeletionQueue;

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	void draw_ui();

	bool upload_mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Mesh& outMesh);

private:
	
	void init_uniform_buffers();

	void init_commands(uint32_t graphics_queue_family);

	void init_sync_structures();

	VkFormat select_depth_format();

	void init_imgui();

	void init_framebuffers(int swapchain_imagecount);

	void init_pipelines();

	void init_depth_image(VkFormat selectedDepthFormat);
};
