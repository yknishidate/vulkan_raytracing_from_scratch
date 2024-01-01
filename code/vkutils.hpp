// DispatchLoaderDynamicをデフォルトディスパッチャとして使うように設定
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

// 1 にするとデバッグメッセージを log.txt に出力する
#define OUTPUT_LOG_FILE 0

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <set>
#include <optional>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>

// デフォルトディスパッチャのためのストレージを用意しておくマクロ
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkutils
{
// 構造体
struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

// 関数宣言
std::vector<const char*> getRequiredExtensions();
bool checkValidationLayerSupport();
bool isDeviceSuitable(const vk::PhysicalDevice&, const vk::SurfaceKHR&);
bool checkDeviceExtensionSupport(const vk::PhysicalDevice&);
void findQueueFamilies(const vk::PhysicalDevice&, const vk::SurfaceKHR&);
vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>&);
vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>&);
vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR&, GLFWwindow*);
SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice, vk::SurfaceKHR);
uint32_t getMemoryType(const vk::MemoryRequirements&, const vk::MemoryPropertyFlags);

// 変数
GLFWwindow* window = nullptr;
vk::PhysicalDevice physicalDevice = nullptr;
uint32_t swapChainImagesSize;
vk::Format swapChainImageFormat;
vk::Extent2D swapChainExtent;
vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
QueueFamilyIndices queueFamilyIndices;
vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{};

bool enableValidationLayers = false;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
    VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
    VK_KHR_MAINTENANCE3_EXTENSION_NAME,
    VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME
};

// 関数定義
#if OUTPUT_LOG_FILE == 0
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                            void* /*pUserData*/)
{
    std::cerr << "messageIDName   = " << pCallbackData->pMessageIdName << "\n";
    for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
        std::cerr << "objectType      = "
            << vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType)) << "\n";
    }
    std::cerr << pCallbackData->pMessage << "\n";
    std::cerr << "\n";
    return VK_FALSE;
}
#else
std::ofstream logFile;
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                            void* /*pUserData*/)
{
    logFile.open("log.txt", std::ios::app);

    logFile << "messageIDName   = " << pCallbackData->pMessageIdName << "\n";
    for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
        logFile << "objectType      = "
            << vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType)) << "\n";
    }
    logFile << pCallbackData->pMessage << "\n";
    logFile << "\n";

    logFile.close();
    return VK_FALSE;
}
#endif

void addDeviceExtensions(const std::vector<const char*>& extensionNames)
{
    for (auto& name : extensionNames) {
        deviceExtensions.push_back(name);
    }
}

void enableDebugMessage()
{
    enableValidationLayers = true;
}

vk::UniqueInstance createInstance()
{
    std::cout << "Create Instance" << std::endl;

    // インスタンスに依存しない関数ポインタを取得する
    static vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo{
        "Application",
        VK_MAKE_VERSION(1, 0, 0),
        "Engine",
        VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_2 };

    auto extensions = getRequiredExtensions();

    vk::UniqueInstance instance;
    if (enableValidationLayers) {
        // デバッグモードの場合
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags{
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError };

        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags{
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
            | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation };

        vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> createInfo{
            { {}, &appInfo, validationLayers, extensions },
            { {}, severityFlags, messageTypeFlags, &debugUtilsMessengerCallback } };
        instance = vk::createInstanceUnique(createInfo.get<vk::InstanceCreateInfo>());
    } else {
        // リリースモードの場合
        vk::InstanceCreateInfo createInfo{ {}, &appInfo, {}, extensions };
        instance = vk::createInstanceUnique(createInfo, nullptr);
    }

    // 全ての関数ポインタを取得する
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    return instance;
}

vk::UniqueDebugUtilsMessengerEXT createDebugMessenger(const vk::Instance& instance)
{
    if (!enableValidationLayers) {
        return vk::UniqueDebugUtilsMessengerEXT{};
    }
    std::cout << "Create Debug Messenger" << std::endl;

    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags{
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError };
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags{
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
        | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
        | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation };

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{
        {}, severityFlags, messageTypeFlags, &debugUtilsMessengerCallback };

    return instance.createDebugUtilsMessengerEXTUnique(createInfo);
}

vk::UniqueSurfaceKHR createSurface(const vk::Instance& instance, GLFWwindow* _window)
{
    std::cout << "Create Surface" << std::endl;

    window = _window;

    // glfw は生の VkSurface や VkInstance で操作する必要がある
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(VkInstance(instance), window, nullptr, &_surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE> _deleter(instance);
    return vk::UniqueSurfaceKHR{ vk::SurfaceKHR(_surface), _deleter };
}

vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface)
{
    // 全ての物理デバイスを取得
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

    // 適切な物理デバイスを選択
    vk::PhysicalDevice physicalDevice;
    for (const auto& device : devices) {
        if (isDeviceSuitable(device, surface)) {
            physicalDevice = device;
            break;
        }
    }

    if (!physicalDevice) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    return physicalDevice;
}

vk::UniqueDevice createLogicalDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface)
{
    std::cout << "Create Logical Device" << std::endl;

    physicalDevice = pickPhysicalDevice(instance, surface);
    physicalDeviceMemoryProperties = physicalDevice.getMemoryProperties();

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{ {}, queueFamilyIndices.graphicsFamily.value(), 1, &queuePriority };

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo
        .setQueueCreateInfos(queueCreateInfo)
        .setPEnabledExtensionNames(deviceExtensions);
    if (enableValidationLayers) {
        deviceCreateInfo.setPEnabledLayerNames(validationLayers);
    }

    vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR> deviceProperties =
        physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    rayTracingPipelineProperties = deviceProperties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    vk::PhysicalDeviceFeatures2 features2 = physicalDevice.getFeatures2();

    vk::StructureChain<
        vk::DeviceCreateInfo,
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceBufferDeviceAddressFeatures> createInfoChain{
            { deviceCreateInfo }, { features2 }, { VK_TRUE }, { VK_TRUE }, { VK_TRUE } };

    vk::UniqueDevice device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    return device;
}

vk::Queue getGraphicsQueue(const vk::Device& device)
{
    return device.getQueue(queueFamilyIndices.graphicsFamily.value(), 0);
}

vk::UniqueSwapchainKHR createSwapChain(const vk::Device& device, const vk::SurfaceKHR& surface)
{
    std::cout << "Create Swap Chain" << std::endl;

    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo
        .setSurface(surface)
        .setMinImageCount(imageCount)
        .setImageFormat(surfaceFormat.format)
        .setImageColorSpace(surfaceFormat.colorSpace)
        .setImageExtent(extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setQueueFamilyIndices(nullptr)
        .setPreTransform(swapChainSupport.capabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(presentMode)
        .setClipped(VK_TRUE)
        .setOldSwapchain(nullptr);

    if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily) {
        std::array indices{ queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.presentFamily.value() };
        createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        createInfo.setQueueFamilyIndices(indices);
    } else {
        createInfo.setQueueFamilyIndices(queueFamilyIndices.graphicsFamily.value());
    }

    vk::UniqueSwapchainKHR swapChain = device.createSwapchainKHRUnique(createInfo);

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    return swapChain;
}

std::vector<vk::Image> getSwapChainImages(const vk::Device& device, const vk::SwapchainKHR& swapChain)
{
    auto images = device.getSwapchainImagesKHR(swapChain);
    swapChainImagesSize = images.size();
    return images;
}

vk::SurfaceFormatKHR getSwapChainImageFormat()
{
    return swapChainImageFormat;
}

vk::Extent2D getSwapChainExtent()
{
    return swapChainExtent;
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Unorm) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eFifoRelaxed) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window)
{
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        vk::Extent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    SwapChainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);

    return details;
}

bool isDeviceSuitable(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
{
    findQueueFamilies(device, surface);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return queueFamilyIndices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device)
{
    std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions{ deviceExtensions.begin(), deviceExtensions.end() };

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    if (!requiredExtensions.empty()) {
        return false;
    }

    std::cout << "Check Device Extension Support: All OK" << std::endl;
    for (auto& extension : deviceExtensions) {
        std::cout << "    " << extension << std::endl;
    }

    return true;
}

void findQueueFamilies(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
{
    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            queueFamilyIndices.graphicsFamily = i;
        }

        VkBool32 presentSupport = device.getSurfaceSupportKHR(i, surface);
        if (presentSupport) {
            queueFamilyIndices.presentFamily = i;
        }

        if (queueFamilyIndices.isComplete()) {
            break;
        }

        i++;
    }
}

std::vector<const char*> getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool checkValidationLayerSupport()
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }
    return true;
}

uint32_t getMemoryType(const vk::MemoryRequirements& memoryRequiriments, const vk::MemoryPropertyFlags memoryProperties)
{
    uint32_t result = -1;
    for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i) {
        if (memoryRequiriments.memoryTypeBits & (1 << i)) {
            if ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & memoryProperties) == memoryProperties) {
                result = i;
                break;
            }
        }
    }
    if (result == -1) {
        throw std::runtime_error("failed to get memory type index.");
    }
    return result;
}

vk::UniqueCommandPool createCommandPool(const vk::Device& device)
{
    return device.createCommandPoolUnique(
        vk::CommandPoolCreateInfo{}
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(queueFamilyIndices.graphicsFamily.value())
    );
}

vk::UniqueCommandBuffer createCommandBuffer(const vk::Device& device, const vk::CommandPool& commandPool, bool begin)
{
    // リストで生成して最初の要素をmoveする
    std::vector<vk::UniqueCommandBuffer> commandBuffers = device.allocateCommandBuffersUnique(
        vk::CommandBufferAllocateInfo{}
        .setCommandPool(commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1)
    );
    vk::UniqueCommandBuffer commandBuffer = std::move(commandBuffers.front());

    if (begin) {
        vk::CommandBufferBeginInfo beginInfo{};
        commandBuffer->begin(&beginInfo);
    }

    return commandBuffer;
}

std::vector<vk::UniqueCommandBuffer> createDrawCommandBuffers(const vk::Device& device, const vk::CommandPool& commandPool)
{
    std::vector<vk::UniqueCommandBuffer> commandBuffers = device.allocateCommandBuffersUnique(
        vk::CommandBufferAllocateInfo{}
        .setCommandPool(commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(swapChainImagesSize)
    );
    return commandBuffers;
}

void submitCommandBuffer(const vk::Device& device, vk::CommandBuffer& commandBuffer, vk::Queue queue)
{
    commandBuffer.end();

    vk::UniqueFence fence = device.createFenceUnique({});

    queue.submit(vk::SubmitInfo{}.setCommandBuffers(commandBuffer), fence.get());

    device.waitForFences(fence.get(), true, std::numeric_limits<uint64_t>::max());
}

std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

vk::UniqueShaderModule createShaderModule(const vk::Device& device, const std::string& filename)
{
    const std::vector<char> code = readFile(filename);

    vk::UniqueShaderModule shaderModule = device.createShaderModuleUnique({
        {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });

    return shaderModule;
}

uint32_t getShaderGroupHandleSize()
{
    return rayTracingPipelineProperties.shaderGroupHandleSize;
}
uint32_t getShaderGroupHandleAlignment()
{
    return rayTracingPipelineProperties.shaderGroupHandleAlignment;
}

inline uint32_t alignedSize(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

void setImageLayout(
    vk::CommandBuffer cmdbuffer,
    vk::Image image,
    vk::ImageLayout oldImageLayout,
    vk::ImageLayout newImageLayout,
    vk::ImageSubresourceRange subresourceRange,
    vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands,
    vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands)
{
    vk::ImageMemoryBarrier imageMemoryBarrier{};
    imageMemoryBarrier
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setOldLayout(oldImageLayout)
        .setNewLayout(newImageLayout)
        .setSubresourceRange(subresourceRange);

    // Source layouts (old)
    switch (oldImageLayout) {
        case vk::ImageLayout::eUndefined:
            imageMemoryBarrier.srcAccessMask = {};
            break;
        case vk::ImageLayout::ePreinitialized:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            break;
    }

    // Target layouts (new)
    switch (newImageLayout) {
        case vk::ImageLayout::eTransferDstOptimal:
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            if (imageMemoryBarrier.srcAccessMask == vk::AccessFlags{}) {
                imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite;
            }
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            break;
    }

    // コマンドバッファにバリアを積む
    cmdbuffer.pipelineBarrier(
        srcStageMask,      // srcStageMask
        dstStageMask,      // dstStageMask
        {},                // dependencyFlags
        {},                // memoryBarriers
        {},                // bufferMemoryBarriers
        imageMemoryBarrier // imageMemoryBarriers
    );
}

uint32_t getHandleSizeAligned()
{
    return alignedSize(getShaderGroupHandleSize(), getShaderGroupHandleAlignment());
}
}
