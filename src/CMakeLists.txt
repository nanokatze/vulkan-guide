
# Add source to this project's executable.
add_executable (vulkan_guide 
"vulkan_guide.cpp"
 "vulkan_guide.h" 
 "vk_mesh.cpp" 
 "vk_mesh.h" 
 "vk_types.h" 
 "vk_initializers.cpp" 
 "vk_initializers.h")


set_property(TARGET vulkan_guide PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vulkan_guide>")

target_include_directories(vulkan_guide PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(vulkan_guide  vkbootstrap vma glm tinyobjloader imgui)

target_link_libraries(vulkan_guide Vulkan::Vulkan sdl2 )