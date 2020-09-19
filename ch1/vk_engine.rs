pub use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::extensions::{ext, khr};
use ash::vk;

use std::borrow::Cow;
use std::ffi::{CStr, CString};

struct VulkanEngineInner {
    graphics_queue_family: u32,
    device: ash::Device,
    graphics_queue: vk::Queue,
    swapchain_loader: khr::Swapchain,
}

impl VulkanEngineInner {
    pub unsafe fn new(
        instance: ash::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        graphics_queue_family: u32,
    ) -> VulkanEngineInner {
        let c_device_extension_names = [khr::Swapchain::name().as_ptr()];
        let features = vk::PhysicalDeviceFeatures::default();
        let priorities = [1.0];
        let queue_info = [
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family)
                .queue_priorities(&priorities)
                .build()
        ];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&c_device_extension_names)
            .enabled_features(&features);
        let device = instance.create_device(physical_device, &device_create_info, None).unwrap();
        let graphics_queue = device.get_device_queue(graphics_queue_family, 0);

        let swapchain_loader = khr::Swapchain::new(&instance, &device);

        VulkanEngineInner{
            graphics_queue_family,
            device,
            graphics_queue,
            swapchain_loader,
        }
    }

    pub unsafe fn draw(&mut self) {
    }
}

impl Drop for VulkanEngineInner {
    fn drop(&mut self) {
    }
}

pub struct VulkanEngine {
    events_loop: winit::EventsLoop,
    window: winit::Window,
    window_extent: vk::Extent2D,

    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ext::DebugUtils,
    debug_callback: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    chosen_gpu: vk::PhysicalDevice,
    graphics_queue_family: u32,

    inner: VulkanEngineInner,
}

impl VulkanEngine {
    pub fn new() -> VulkanEngine {
        let events_loop = winit::EventsLoop::new();
        let window_extent = vk::Extent2D{width: 1700, height: 900};
        let window = winit::WindowBuilder::new()
            .with_title("VulkanEngine")
            .with_dimensions(winit::dpi::LogicalSize::new(
                f64::from(window_extent.width),
                f64::from(window_extent.height),
            ))
            .build(&events_loop).unwrap();

        unsafe {
            let entry = ash::Entry::new().unwrap();

            let app_name = CString::new("Example Vulkan Application").unwrap();
            let appinfo = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::make_version(1, 0, 0));
            let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
            let c_layer_names: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();
            let surface_extensions = ash_window::enumerate_required_extensions(&window).unwrap();
            let mut c_extension_names = surface_extensions.iter().map(|ext| ext.as_ptr()).collect::<Vec<_>>();
            c_extension_names.push(ext::DebugUtils::name().as_ptr());
            let instance_create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&c_layer_names)
                .enabled_extension_names(&c_extension_names);
            let instance = entry.create_instance(&instance_create_info, None).unwrap();

            let debug_utils_loader = ext::DebugUtils::new(&entry, &instance);

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(debug_messenger));
            let debug_callback = debug_utils_loader.create_debug_utils_messenger(&debug_info, None).unwrap();

            let surface = ash_window::create_surface(&entry, &instance, &window, None).unwrap();

            let surface_loader = khr::Surface::new(&entry, &instance);
            let (chosen_gpu, graphics_queue_family) = instance
                .enumerate_physical_devices().unwrap()
                .iter()
                .map(|p_physical_device| {
                    instance
                        .get_physical_device_queue_family_properties(*p_physical_device)
                        .iter()
                        .enumerate()
                        .filter_map(|(i, ref info)| {
                            let queue_family_index = i as u32;
                            let supports_graphics = info.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                            let supports_surface = surface_loader
                                .get_physical_device_surface_support(
                                    *p_physical_device,
                                    queue_family_index,
                                    surface).unwrap();

                            if supports_graphics && supports_surface {
                                Some((*p_physical_device, queue_family_index))
                            } else {
                                None
                            }
                        })
                        .next()
                })
                .filter_map(|v| v)
                .next().unwrap();

            let inner = VulkanEngineInner::new(instance.clone(), surface.clone(), chosen_gpu.clone(), graphics_queue_family);

            VulkanEngine{
                events_loop: events_loop,
                window_extent,
                window,

                entry,
                instance,
                debug_utils_loader,
                debug_callback,
                surface,
                chosen_gpu,
                graphics_queue_family,

                inner,
            }
        }
    }

    pub fn run(&mut self) {
        use winit::*;

        let inner = &mut self.inner;

        self.events_loop.run_forever(|event| {
            let cf = match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => winit::ControlFlow::Break,
                    _ => ControlFlow::Continue,
                },
                _ => ControlFlow::Continue,
            };
            unsafe { inner.draw() };
            cf
        });
    }
}

unsafe extern "system" fn debug_messenger(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}
