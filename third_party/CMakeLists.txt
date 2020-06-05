find_package(Vulkan REQUIRED)

add_library(volk STATIC)
add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(vma INTERFACE)
add_library(tinyobjloader STATIC)

target_sources(volk PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/volk/volk.h" 
    "${CMAKE_CURRENT_SOURCE_DIR}/volk/volk.c"
)
target_include_directories(volk PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/volk" )

target_link_libraries(volk PUBLIC Vulkan::Vulkan)

target_sources(vkbootstrap PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/vkbootstrap/VkBootstrap.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/vkbootstrap/VkBootstrap.cpp"
)

target_include_directories(vkbootstrap PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/vkbootstrap" )
target_link_libraries(vkbootstrap PUBLIC volk)

#both vma and glm and header only libs so we only need the include path
target_include_directories(vma INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/vma" )

target_include_directories(glm INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm" )

target_sources(tinyobjloader PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader/tiny_obj_loader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader/tiny_obj_loader.cc"
)

target_include_directories(tinyobjloader PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader" )