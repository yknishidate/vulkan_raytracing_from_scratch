#pragma once
#include "vkutils.hpp"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

struct Vertex {
    float pos[3];
};

struct Buffer {
    vk::UniqueBuffer handle;
    vk::UniqueDeviceMemory deviceMemory;
    uint64_t deviceAddress;
};

struct AccelStruct {
    vk::UniqueAccelerationStructureKHR handle;
    Buffer buffer;
};

class Application {
public:
    void run() {
        initWindow();
        initVulkan();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    }

private:
    GLFWwindow* window = nullptr;

    // Instance, Device, Queue
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT debugMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;
    vk::Queue queue;
    uint32_t queueFamilyIndex{};

    // Command buffer
    vk::UniqueCommandPool commandPool;
    vk::UniqueCommandBuffer commandBuffer;

    // Swapchain
    vk::SurfaceFormatKHR surfaceFormat;
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueImageView> swapchainImageViews;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        std::vector<const char*> layers = {
            "VK_LAYER_KHRONOS_validation",
        };

        std::vector<const char*> deviceExtensions = {
            // For swapchain
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            // For ray tracing
            VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        };

        // Create instance, device, queue
        // Ray tracing requires Vulkan 1.2 or later
        instance = vkutils::createInstance(VK_API_VERSION_1_2, layers);
        debugMessenger = vkutils::createDebugMessenger(*instance);
        surface = vkutils::createSurface(*instance, window);
        physicalDevice =
            vkutils::pickPhysicalDevice(*instance, *surface, deviceExtensions);
        queueFamilyIndex =
            vkutils::findGeneralQueueFamily(physicalDevice, *surface);
        device = vkutils::createLogicalDevice(physicalDevice, queueFamilyIndex,
                                              deviceExtensions);
        queue = device->getQueue(queueFamilyIndex, 0);

        // Create command buffers
        commandPool = vkutils::createCommandPool(*device, queueFamilyIndex);
        commandBuffer = vkutils::createCommandBuffer(*device, *commandPool);

        // Create swapchain
        // Specify images as storage images
        surfaceFormat = vkutils::chooseSurfaceFormat(physicalDevice, *surface);
        swapchain = vkutils::createSwapchain(
            physicalDevice, *device, *surface, queueFamilyIndex,
            vk::ImageUsageFlagBits::eStorage, surfaceFormat, WIDTH, HEIGHT);
        swapchainImages = device->getSwapchainImagesKHR(*swapchain);
        createSwapchainImageViews();
    }

    void createSwapchainImageViews() {
        for (auto& image : swapchainImages) {
            vk::ImageViewCreateInfo imageViewCreateInfo{};
            imageViewCreateInfo.setImage(image);
            imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
            imageViewCreateInfo.setFormat(surfaceFormat.format);
            imageViewCreateInfo.setComponents(
                {vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                 vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
            imageViewCreateInfo.setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            swapchainImageViews.push_back(
                device->createImageViewUnique(imageViewCreateInfo));
        }

        vkutils::oneTimeSubmit(
            *device, *commandPool, queue, [&](vk::CommandBuffer commandBuffer) {
                for (auto& image : swapchainImages) {
                    vkutils::setImageLayout(
                        commandBuffer, image, vk::ImageLayout::eUndefined,
                        vk::ImageLayout::ePresentSrcKHR,
                        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
                }
            });
    }
};
