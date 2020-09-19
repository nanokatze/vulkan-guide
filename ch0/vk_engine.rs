use ash::vk;

struct VulkanEngineInner {
}

impl VulkanEngineInner {
    pub fn new() -> VulkanEngineInner {
        return VulkanEngineInner{}
    }

    pub fn draw(&mut self) {
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
            .build(&events_loop)
            .unwrap();
        VulkanEngine{
            events_loop: events_loop,
            window_extent,
            window,
            inner: VulkanEngineInner::new(),
        }
    }

    pub fn run(&mut self) {
        use winit::*;

        let mut inner = &mut self.inner;

        self.events_loop.run_forever(|event| {
            let cf = match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => winit::ControlFlow::Break,
                    _ => ControlFlow::Continue,
                },
                _ => ControlFlow::Continue,
            };
            inner.draw();
            cf
        });
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
    }
}