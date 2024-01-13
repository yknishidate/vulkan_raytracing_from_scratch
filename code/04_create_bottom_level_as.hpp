#pragma once
#include "vkutils.hpp"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

struct Buffer {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DeviceAddress address{};

    void init(vk::PhysicalDevice physicalDevice,
              vk::Device device,
              vk::DeviceSize size,
              vk::BufferUsageFlags usage,
              vk::MemoryPropertyFlags memoryProperty,
              const void* data = nullptr) {
        // Create buffer
        vk::BufferCreateInfo createInfo{};
        createInfo.setSize(size);
        createInfo.setUsage(usage);
        buffer = device.createBufferUnique(createInfo);

        // Allocate memory
        vk::MemoryRequirements memoryReq =
            device.getBufferMemoryRequirements(*buffer);
        vk::MemoryAllocateFlagsInfo allocateFlags{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            allocateFlags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        }

        uint32_t memoryType = vkutils::getMemoryType(physicalDevice,  //
                                                     memoryReq, memoryProperty);
        vk::MemoryAllocateInfo allocateInfo{};
        allocateInfo.setAllocationSize(memoryReq.size);
        allocateInfo.setMemoryTypeIndex(memoryType);
        allocateInfo.setPNext(&allocateFlags);
        memory = device.allocateMemoryUnique(allocateInfo);

        // Bind buffer to memory
        device.bindBufferMemory(*buffer, *memory, 0);

        // Copy data
        if (data) {
            void* mappedPtr = device.mapMemory(*memory, 0, size);
            memcpy(mappedPtr, data, size);
            device.unmapMemory(*memory);
        }

        // Get address
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            vk::BufferDeviceAddressInfoKHR addressInfo{};
            addressInfo.setBuffer(*buffer);
            address = device.getBufferAddressKHR(&addressInfo);
        }
    }
};

struct Vertex {
    float pos[3];
};

struct AccelStruct {
    vk::UniqueAccelerationStructureKHR accel;
    Buffer buffer;

    void init(vk::PhysicalDevice physicalDevice,
              vk::Device device,
              vk::CommandPool commandPool,
              vk::Queue queue,
              vk::AccelerationStructureTypeKHR type,
              vk::AccelerationStructureGeometryKHR geometry,
              uint32_t primitiveCount) {
        // Get build info
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.setType(type);
        buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
        buildInfo.setFlags(
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildInfo.setGeometries(geometry);

        vk::AccelerationStructureBuildSizesInfoKHR buildSizes =
            device.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
                primitiveCount);

        // Create buffer for AS
        buffer.init(physicalDevice, device,
                    buildSizes.accelerationStructureSize,
                    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Create AS
        vk::AccelerationStructureCreateInfoKHR createInfo{};
        createInfo.setBuffer(*buffer.buffer);
        createInfo.setSize(buildSizes.accelerationStructureSize);
        createInfo.setType(type);
        accel = device.createAccelerationStructureKHRUnique(createInfo);

        // Create scratch buffer
        Buffer scratchBuffer;
        scratchBuffer.init(physicalDevice, device, buildSizes.buildScratchSize,
                           vk::BufferUsageFlagBits::eStorageBuffer |
                               vk::BufferUsageFlagBits::eShaderDeviceAddress,
                           vk::MemoryPropertyFlagBits::eDeviceLocal);

        buildInfo.setDstAccelerationStructure(*accel);
        buildInfo.setScratchData(scratchBuffer.address);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.setPrimitiveCount(primitiveCount);
        buildRangeInfo.setPrimitiveOffset(0);
        buildRangeInfo.setFirstVertex(0);
        buildRangeInfo.setTransformOffset(0);

        // Build
        vkutils::oneTimeSubmit(          //
            device, commandPool, queue,  //
            [&](vk::CommandBuffer commandBuffer) {
                commandBuffer.buildAccelerationStructuresKHR(buildInfo,
                                                             &buildRangeInfo);
            });

        // Get address
        vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.setAccelerationStructure(*accel);
        buffer.address = device.getAccelerationStructureAddressKHR(addressInfo);
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

    // Acceleration structure
    AccelStruct bottomAccel{};

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
        createBottomLevelAS();
    }

    void createSwapchainImageViews() {
        for (auto image : swapchainImages) {
            vk::ImageViewCreateInfo createInfo{};
            createInfo.setImage(image);
            createInfo.setViewType(vk::ImageViewType::e2D);
            createInfo.setFormat(surfaceFormat.format);
            createInfo.setComponents(
                {vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                 vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
            createInfo.setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            swapchainImageViews.push_back(
                device->createImageViewUnique(createInfo));
        }

        vkutils::oneTimeSubmit(
            *device, *commandPool, queue, [&](vk::CommandBuffer commandBuffer) {
                for (auto image : swapchainImages) {
                    vkutils::setImageLayout(
                        commandBuffer, image,  //
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::ePresentSrcKHR,
                        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
                }
            });
    }

    void createBottomLevelAS() {
        std::cout << "Create BLAS\n";

        // Prepare a triangle data
        std::vector<Vertex> vertices = {
            {{1.0f, 1.0f, 0.0f}},
            {{-1.0f, 1.0f, 0.0f}},
            {{0.0f, -1.0f, 0.0f}},
        };
        std::vector<uint32_t> indices = {0, 1, 2};

        // Create vertex buffer and index buffer
        vk::BufferUsageFlags bufferUsage{
            vk::BufferUsageFlagBits::
                eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress};
        vk::MemoryPropertyFlags memoryProperty{
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent};
        Buffer vertexBuffer;
        Buffer indexBuffer;
        vertexBuffer.init(physicalDevice, *device,           //
                          vertices.size() * sizeof(Vertex),  //
                          bufferUsage, memoryProperty, vertices.data());
        indexBuffer.init(physicalDevice, *device,            //
                         indices.size() * sizeof(uint32_t),  //
                         bufferUsage, memoryProperty, indices.data());

        // Create geometry
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangles.setVertexData(vertexBuffer.address);
        triangles.setVertexStride(sizeof(Vertex));
        triangles.setMaxVertex(static_cast<uint32_t>(vertices.size()));
        triangles.setIndexType(vk::IndexType::eUint32);
        triangles.setIndexData(indexBuffer.address);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometry.setGeometry({triangles});
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // Create and build BLAS
        uint32_t primitiveCount = static_cast<uint32_t>(indices.size() / 3);
        bottomAccel.init(physicalDevice, *device, *commandPool, queue,
                         vk::AccelerationStructureTypeKHR::eBottomLevel,
                         geometry, primitiveCount);
    }
};
