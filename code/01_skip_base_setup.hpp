#pragma once
#include "vkutils.hpp"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

class Application
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueDevice device;
    vk::Queue graphicsQueue;

    vk::UniqueSwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> drawCommandBuffers;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan()
    {
        std::vector<const char*> deviceExtensions = {
            // レイトレーシング拡張
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME
        };
        vkutils::addDeviceExtensions(deviceExtensions);

        vkutils::enableDebugMessage();

        instance = vkutils::createInstance();
        debugUtilsMessenger = vkutils::createDebugMessenger(instance.get());
        surface = vkutils::createSurface(instance.get(), window);
        device = vkutils::createLogicalDevice(instance.get(), surface.get());
        graphicsQueue = vkutils::getGraphicsQueue(device.get());

        swapChain = vkutils::createSwapChain(device.get(), surface.get());
        swapChainImages = vkutils::getSwapChainImages(device.get(), swapChain.get());

        commandPool = vkutils::createCommandPool(device.get());
        drawCommandBuffers = vkutils::createDrawCommandBuffers(device.get(), commandPool.get());
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};
