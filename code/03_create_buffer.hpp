#pragma once
#include "vkutils.hpp"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

struct Buffer {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory deviceMemory;
    uint64_t deviceAddress{};

    void init(vk::PhysicalDevice physicalDevice,
              vk::Device device,
              vk::DeviceSize size,
              vk::BufferUsageFlags usage,
              vk::MemoryPropertyFlags memoryProperty,
              const void* data = nullptr) {
        // Create buffer
        vk::BufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.setSize(size);
        bufferCreateInfo.setUsage(usage);
        bufferCreateInfo.setQueueFamilyIndexCount(0);
        buffer = device.createBufferUnique(bufferCreateInfo);

        // Allocate memory
        auto memoryRequirements = device.getBufferMemoryRequirements(*buffer);
        vk::MemoryAllocateFlagsInfo memoryFlagsInfo{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            memoryFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        }

        vk::MemoryAllocateInfo allocateInfo{};
        allocateInfo.setAllocationSize(memoryRequirements.size);
        allocateInfo.setMemoryTypeIndex(vkutils::getMemoryType(
            physicalDevice, memoryRequirements, memoryProperty));
        allocateInfo.setPNext(&memoryFlagsInfo);
        deviceMemory = device.allocateMemoryUnique(allocateInfo);

        // Bind buffer to memory
        device.bindBufferMemory(*buffer, *deviceMemory, 0);

        // Copy data
        if (data) {
            void* dataPtr = device.mapMemory(*deviceMemory, 0, size);
            memcpy(dataPtr, data, size);
            device.unmapMemory(*deviceMemory);
        }

        // Get address
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            vk::BufferDeviceAddressInfoKHR bufferDeviceAI{*buffer};
            deviceAddress = device.getBufferAddressKHR(&bufferDeviceAI);
        }
    }
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
                        commandBuffer, image,  //
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::ePresentSrcKHR,
                        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
                }
            });
    }
};
