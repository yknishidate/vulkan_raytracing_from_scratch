
#include "vkutils.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

struct StorageImage
{
    vk::UniqueDeviceMemory memory;
    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::Format format;
    uint32_t width;
    uint32_t height;
};

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
    std::vector<vk::UniqueImageView> swapChainImageViews;

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> drawCommandBuffers;

    StorageImage storageImage;

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
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,

            // VK_KHR_acceleration_structure のために必要
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME
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
        swapChainImageViews = vkutils::createImageViews(device.get(), swapChainImages);

        commandPool = vkutils::createCommandPool(device.get());
        drawCommandBuffers = vkutils::createDrawCommandBuffers(device.get(), commandPool.get());

        createStorageImage();
    }

    void createStorageImage()
    {
        storageImage.width = WIDTH;
        storageImage.height = HEIGHT;

        // Imageハンドルを作成する
        storageImage.image = device->createImageUnique(
            vk::ImageCreateInfo{}
            .setImageType(vk::ImageType::e2D)
            .setFormat(vk::Format::eB8G8R8A8Unorm)
            .setExtent({ storageImage.width , storageImage.height, 1 })
            .setMipLevels(1)
            .setArrayLayers(1)
            .setUsage(vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage)
        );

        // メモリ確保を行いバインドする
        auto memoryRequirements = device->getImageMemoryRequirements(storageImage.image.get());
        storageImage.memory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
            .setAllocationSize(memoryRequirements.size)
            .setMemoryTypeIndex(vkutils::getMemoryType(
                memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal))
        );
        device->bindImageMemory(storageImage.image.get(), storageImage.memory.get(), 0);

        // Image Viewを作成する
        storageImage.view = device->createImageViewUnique(
            vk::ImageViewCreateInfo{}
            .setImage(storageImage.image.get())
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(vk::Format::eB8G8R8A8Unorm)
            .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 })
        );

        // Image レイアウトをGeneralにしておく
        auto commandBuffer = vkutils::createCommandBuffer(device.get(), commandPool.get(), true);

        vkutils::setImageLayout(commandBuffer.get(), storageImage.image.get(),
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

        vkutils::submitCommandBuffer(device.get(), commandBuffer.get(), graphicsQueue);
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

int main()
{
    Application app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
