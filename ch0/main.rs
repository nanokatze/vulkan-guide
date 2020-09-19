mod vk_engine;
mod vk_initializers;
mod vk_types;

use vk_engine::*;

fn main() {
    let mut engine = VulkanEngine::new();
    engine.run();
}
