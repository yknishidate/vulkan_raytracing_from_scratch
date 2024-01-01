#pragma once

// DispatchLoaderDynamicをデフォルトディスパッチャとして使うように設定
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

// デフォルトディスパッチャのためのストレージを用意しておくマクロ
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkutils {
// 関数定義
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                            void* pUserData) {
    std::cerr << pCallbackData->pMessage << "\n\n";
    return VK_FALSE;
}

inline bool checkLayerSupport(const std::vector<const char*>& layers) {
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const char* layerName : layers) {
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

inline std::vector<const char*> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    return extensions;
}

inline vk::UniqueInstance createInstance(const std::vector<const char*>& layers) {
    std::cout << "Create instance\n";

    // インスタンスに依存しない関数ポインタを取得する
    static vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    if (!checkLayerSupport(layers)) {
        std::cerr << "Requested layers not available.\n";
        std::abort();
    }

    vk::ApplicationInfo appInfo{};
    appInfo.setApiVersion(VK_API_VERSION_1_2);

    auto extensions = getRequiredExtensions();

    // デバッグモードの場合
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags{
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError};

    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags{
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation};

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> createInfo{
        {{}, &appInfo, layers, extensions},
        {{}, severityFlags, messageTypeFlags, &debugUtilsMessengerCallback},
    };

    vk::UniqueInstance instance =
        vk::createInstanceUnique(createInfo.get<vk::InstanceCreateInfo>());

    // 全ての関数ポインタを取得する
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    return instance;
}

inline vk::UniqueDebugUtilsMessengerEXT createDebugMessenger(vk::Instance instance) {
    std::cout << "Create debug messenger\n";

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    createInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                              vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                              vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
    createInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
    return instance.createDebugUtilsMessengerEXTUnique(createInfo);
}

inline vk::UniqueSurfaceKHR createSurface(vk::Instance instance, GLFWwindow* window) {
    std::cout << "Create surface\n";

    // glfw は生の VkSurface で操作する必要がある
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &_surface) != VK_SUCCESS) {
        std::cerr << "Failed to create window surface.\n";
        std::abort();
    }
    return vk::UniqueSurfaceKHR{vk::SurfaceKHR(_surface), {instance}};
}

inline uint32_t findGeneralQueueFamilies(vk::PhysicalDevice physicalDevice,
                                         vk::SurfaceKHR surface) {
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        vk::Bool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, surface);
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics && presentSupport) {
            return i;
        }
    }
    std::cerr << "Failed to find general queue family.\n";
    std::abort();
}

inline bool checkDeviceExtensionSupport(vk::PhysicalDevice device,
                                        const std::vector<const char*>& deviceExtensions) {
    std::set<std::string> requiredExtensions{deviceExtensions.begin(), deviceExtensions.end()};
    for (const auto& extension : device.enumerateDeviceExtensionProperties()) {
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}

inline bool isDeviceSuitable(vk::PhysicalDevice device,
                             vk::SurfaceKHR surface,
                             const std::vector<const char*>& deviceExtensions) {
    bool extensionsSupported = checkDeviceExtensionSupport(device, deviceExtensions);
    if (!extensionsSupported) {
        return false;
    }

    std::vector<vk::SurfaceFormatKHR> formats = device.getSurfaceFormatsKHR(surface);
    std::vector<vk::PresentModeKHR> presentModes = device.getSurfacePresentModesKHR(surface);
    return !formats.empty() && !presentModes.empty();
}

inline vk::PhysicalDevice pickPhysicalDevice(vk::Instance instance,
                                             vk::SurfaceKHR surface,
                                             const std::vector<const char*>& deviceExtensions) {
    // 適切な物理デバイスを選択
    for (const auto& device : instance.enumeratePhysicalDevices()) {
        if (isDeviceSuitable(device, surface, deviceExtensions)) {
            return device;
        }
    }

    std::cerr << "Failed to find physical device.\n";
    std::abort();
}

inline auto getRayTracingProps(vk::PhysicalDevice physicalDevice) {
    auto deviceProperties =
        physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                      vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    return deviceProperties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
}

inline vk::UniqueDevice createLogicalDevice(vk::PhysicalDevice physicalDevice,
                                            uint32_t queueFamilyIndex,
                                            const std::vector<const char*>& deviceExtensions) {
    std::cout << "Create device\n";

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{{}, queueFamilyIndex, 1, &queuePriority};

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.setQueueCreateInfos(queueCreateInfo);
    deviceCreateInfo.setPEnabledExtensionNames(deviceExtensions);

    vk::PhysicalDeviceFeatures2 features2 = physicalDevice.getFeatures2();

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
                       vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                       vk::PhysicalDeviceBufferDeviceAddressFeatures>
        createInfoChain{{deviceCreateInfo}, {features2}, {VK_TRUE}, {VK_TRUE}, {VK_TRUE}};

    vk::UniqueDevice device =
        physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    return device;
}

inline vk::SurfaceFormatKHR chooseSwapSurfaceFormat(vk::PhysicalDevice physicalDevice,
                                                    vk::SurfaceKHR surface) {
    auto availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Unorm) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

inline vk::PresentModeKHR chooseSwapPresentMode(vk::PhysicalDevice physicalDevice,
                                                vk::SurfaceKHR surface) {
    auto availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eFifoRelaxed) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

inline vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR capabilities,
                                     uint32_t width,
                                     uint32_t height) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        vk::Extent2D actualExtent = {width, height};
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);
        return actualExtent;
    }
}

inline vk::UniqueSwapchainKHR createSwapchain(vk::PhysicalDevice physicalDevice,
                                              vk::Device device,
                                              vk::SurfaceKHR surface,
                                              uint32_t queueFamilyIndex,
                                              uint32_t width,
                                              uint32_t height) {
    std::cout << "Create swapchain\n";

    vk::SurfaceCapabilitiesKHR capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(physicalDevice, surface);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(physicalDevice, surface);
    vk::Extent2D extent = chooseSwapExtent(capabilities, width, height);

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.setSurface(surface);
    createInfo.setMinImageCount(imageCount);
    createInfo.setImageFormat(surfaceFormat.format);
    createInfo.setImageColorSpace(surfaceFormat.colorSpace);
    createInfo.setImageExtent(extent);
    createInfo.setImageArrayLayers(1);
    createInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst |
                             vk::ImageUsageFlagBits::eStorage |
                             vk::ImageUsageFlagBits::eTransferSrc);  // TODO: remove
    createInfo.setQueueFamilyIndices(nullptr);
    createInfo.setPreTransform(capabilities.currentTransform);
    createInfo.setPresentMode(presentMode);
    createInfo.setClipped(VK_TRUE);
    createInfo.setQueueFamilyIndices(queueFamilyIndex);
    return device.createSwapchainKHRUnique(createInfo);
}

inline uint32_t getMemoryType(vk::PhysicalDevice physicalDevice,
                              vk::MemoryRequirements memoryRequirements,
                              vk::MemoryPropertyFlags memoryProperties) {
    auto physicalDeviceMemoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i) {
        if (memoryRequirements.memoryTypeBits & (1 << i)) {
            if ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & memoryProperties) ==
                memoryProperties) {
                return i;
            }
        }
    }

    std::cerr << "Failed to get memory type index.\n";
    std::abort();
}

inline vk::UniqueCommandPool createCommandPool(vk::Device device, uint32_t queueFamilyIndex) {
    vk::CommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolCreateInfo.setQueueFamilyIndex(queueFamilyIndex);
    return device.createCommandPoolUnique(commandPoolCreateInfo);
}

inline void oneTimeSubmit(vk::Device device,
                          vk::CommandPool commandPool,
                          vk::Queue queue,
                          const std::function<void(vk::CommandBuffer)>& func) {
    // Allocate
    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.setCommandPool(commandPool);
    allocateInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocateInfo.setCommandBufferCount(1);
    auto commandBuffers = device.allocateCommandBuffersUnique(allocateInfo);

    // Record
    commandBuffers[0]->begin(vk::CommandBufferBeginInfo{});
    func(commandBuffers[0].get());
    commandBuffers[0]->end();

    // Submit
    vk::UniqueFence fence = device.createFenceUnique({});
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(commandBuffers[0].get());
    queue.submit(submitInfo, fence.get());

    // Wait
    if (device.waitForFences(fence.get(), true, std::numeric_limits<uint64_t>::max()) !=
        vk::Result::eSuccess) {
        std::cerr << "Failed to wait for fence.\n";
        std::abort();
    }
}

inline std::vector<vk::UniqueCommandBuffer> createDrawCommandBuffers(vk::Device device,
                                                                     vk::CommandPool commandPool,
                                                                     uint32_t count) {
    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.setCommandPool(commandPool);
    allocateInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocateInfo.setCommandBufferCount(count);
    return device.allocateCommandBuffersUnique(allocateInfo);
}

inline std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file!\n";
        std::abort();
    }

    size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

inline vk::UniqueShaderModule createShaderModule(vk::Device device, const std::string& filename) {
    const std::vector<char> code = readFile(filename);

    vk::UniqueShaderModule shaderModule = device.createShaderModuleUnique(
        {{}, code.size(), reinterpret_cast<const uint32_t*>(code.data())});

    return shaderModule;
}

inline void setImageLayout(
    vk::CommandBuffer commandBuffer,
    vk::Image image,
    vk::ImageLayout oldImageLayout,
    vk::ImageLayout newImageLayout,
    vk::ImageSubresourceRange subresourceRange,
    vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands,
    vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands) {
    vk::ImageMemoryBarrier imageMemoryBarrier{};
    imageMemoryBarrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    imageMemoryBarrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    imageMemoryBarrier.setImage(image);
    imageMemoryBarrier.setOldLayout(oldImageLayout);
    imageMemoryBarrier.setNewLayout(newImageLayout);
    imageMemoryBarrier.setSubresourceRange(subresourceRange);

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
            imageMemoryBarrier.dstAccessMask =
                imageMemoryBarrier.dstAccessMask | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            if (imageMemoryBarrier.srcAccessMask == vk::AccessFlags{}) {
                imageMemoryBarrier.srcAccessMask =
                    vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite;
            }
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            break;
    }

    // コマンドバッファにバリアを積む
    commandBuffer.pipelineBarrier(srcStageMask, dstStageMask,  //
                                  {}, {}, {}, imageMemoryBarrier);
}

inline uint32_t getHandleSizeAligned(vk::PhysicalDeviceRayTracingPipelinePropertiesKHR props) {
    const uint32_t handleSize = props.shaderGroupHandleSize;
    const uint32_t handleAlignment = props.shaderGroupHandleAlignment;
    return (handleSize + handleAlignment - 1) & ~(handleAlignment - 1);
}
}  // namespace vkutils
