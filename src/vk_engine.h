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

class VulkanEngine {
public:

	VmaAllocator _allocator;
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

	Mesh _monkeyMesh;

	uint64_t _frameNumber;
	bool _isInitialized = false;

	bool _drawFunky = false;

	glm::vec3 camPos;

	DeletionQueue _mainDeletionQueue;

	void init();
	void cleanup();
	void draw();

	bool upload_mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Mesh& outMesh);
};
