#include <vk_engine.h>

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>

#include <SDL.h>
#include <SDL_vulkan.h>

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
