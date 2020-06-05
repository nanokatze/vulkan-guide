add_library(volk INTERFACE)
add_library(vkbootstrap INTERFACE)
add_library(glm INTERFACE)
add_library(vma INTERFACE)

target_sources(volk INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/volk/volk.h" 
    "${CMAKE_CURRENT_SOURCE_DIR}/volk/volk.c"
)
target_include_directories(volk INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/volk" )

target_sources(vkbootstrap INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/vkbootstrap/VkBootstrap.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/vkbootstrap/VkBootstrap.cpp"
)

target_include_directories(vkbootstrap INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/vkbootstrap" )


#both vma and glm and header only libs so we only need the include path
target_include_directories(vma INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/vma" )

target_include_directories(glm INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm" )