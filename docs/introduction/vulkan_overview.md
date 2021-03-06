---
layout: default
title: Vulkan API
nav_order: 1
parent: introduction
---

![VulkanLogo](/assets/images/Vulkan_170px_Dec16.jpg)

## What is Vulkan?

Vulkan is a multiplatform API (application program interface) created as a successor to OpenGl by the same authors at the Khronos committee.
Vulkan is developed by a committee of GPU vendors and other partners, and the API is a standard that anyone can implement.

Vulkan cuts the design decisions of OpenGL, and starts brand new. It is designed for multithreaded high performance applications, and its much more explicit API is made so that drivers have much less guesswork to do. If used right, Vulkan can provide a really high level of performance.


## When to use Vulkan

While Vulkan is the newest API, you should not be using Vulkan if you dont need its features.
Writing a Vulkan engine is an order of magnitude harder than using Opengl, and the GPU-side performance will not necessarily be better.

The main thing Vulkan improves is CPU overheads and multithreading, and control over GPU hitches.
When Vulkan is used right, it is possible to almost completely eliminate hitches, achieving a very smooth framerate, on very low CPU overhead.

If your target application is a GPU bound application, Vulkan is not likely to improve things in a way that its worth the extra complexity.

If you want to make big maps and dynamic worlds, or CAD type scenes with a lot of objects, the multithread capabilities of Vulkan will likely be very useful.

On the mobile market, there is the problem of fragmentation. Not every phone implements Vulkan (old phones dont), and of those that do, you might run into problems with driver quality. On the phones that implement Vulkan decently well, the lowered CPU overhead and the better map to the hardware will mean you can improve the performance and battery usage of the application extensively.


# Overview of Vulkan API
Vulkan is an API, which means its essentially just a list of functions to call, and parameters to those.
It is a spec based on the C language, so it can be used from almost every language. While this tutorial is written in C++, what is done here can be relatively easily done from other low level languages such as Rust or just plain C.

## Multiplatform

Vulkan is uniquely designed compared to other GPU APIs, in that its the same API for both Mobile GPUs, and Desktop CPUs. Almost every feature in Vulkan is an optional feature, because phones dont support it, while PC GPUs might. In this tutorial, we are focusing on PC, so we are going to do things that wont directly run in smartphones and tablets.

If you are looking to have an application that will run on both PC and smartphones, its recomended that you have 2 core render paths. As the API its the same, a lot of code can be shared directly, but the differences in features and fast-paths between the 2 targets will mean that if you try to have just one render code, it will be suboptimal on one of the two. 

During this guide we will explain the things that are most different between the 2 types of GPUs.

## Validation layers

There is a significant amount of things in the Vulkan API that you have to do, but they actually do nothing on the real hardware. This means that even on some wrong code, it can still run just fine on some hardware, but break spectacularly on others. This is specially bad on syncronization code, which is very hard to get right, and it can run very different depending on the hardware unless you do everything according to the spec. Something that runs fine on a Nvidia card might break on an AMD card, and its the same in reverse.

Those are some of the main reasons Vulkan comes with a set of "Validation Layers". They are hooked between the API calls from your code, and the actual gpu driver, and they check that you are doing the correct things according to the Vulkan spec. While developing, its imperative to use them to make sure you arent doing the wrong things. They can be enabled and disabled at startup, so when profiling or releasing the aplication, you would usually disable them. They do bring a significant hit to performance due to all that checking. 

## Usage and general mindset
In the Vulkan API, almost everything is designed around objects that you create manually and then use. This is not only for the actual gpu resources such as Images/Textures and Buffers (for memory or things like Vertex data). But also for a lot of "internal" configuration structures.

For example, things like gpu draw parameters are stored into a Pipeline object, holding shaders and other configuration. In Opengl and DX11, this is calculated "on the fly" while rendering.
When  you use Vulkan, you need to think if its worth to cache these objects, or create them while rendering on-the-fly. Some objects like Pipelines are very expensive to create, so its best to create them on load screens or background threads. Other objects are cheaper to create, such as DescriptorSets, so its fine to create them at runtime.

Because everything in Vulkan is "pre-built" by default, it means that most of the state validation in the GPU will be done when you create the object, and the rendering itself does less work and is faster. Good understanding of how these objects are created and used will allow you to control how everything executes in a way that will make the framerate smooth.

When doing actual GPU commands, all of the work has to be done on a CommandBuffer, and submitted into a Queue. You first allocate a command buffer, then start encoding things on it, and then you execute it by adding it into a Queue. When you submit a command buffer into a queue, it will immediately start executing on the GPU side. You have tools to control when that execution has finished. If you submit multiple command buffers into different queues, it is possible that they execute in parallel. 

There is no concept of frames in Vulkan. This means that the way you render is entirely up to you. The only thing that matters is when you have to display the frame to the screen, which is done through a swapchain. But there is no fundamental difference between rendering and then sending the images over the network, or saving the images into a file, or displaying it into the screen through the swapchain.

Its possible to use Vulkan in an entirely headless mode, not even displaying anything to the user. You can render to images and then store them into disk (very useful for testing!) or  use Vulkan as a way to perform GPU calculations such as a raytracer.

Next: [Vulkan Render flow]({{ site.baseurl }}{% link docs/introduction/vulkan_execution.md %})