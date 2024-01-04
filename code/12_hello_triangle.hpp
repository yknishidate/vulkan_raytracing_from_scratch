#pragma once
#include "vkutils.hpp"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

struct Vertex {
    float pos[3];
};

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
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo.setType(type);
        buildGeometryInfo.setFlags(
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildGeometryInfo.setGeometries(geometry);
        auto buildSizesInfo = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo,
            primitiveCount);

        // Create buffer for AS
        buffer.init(physicalDevice, device,
                    buildSizesInfo.accelerationStructureSize,
                    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Create AS
        vk::AccelerationStructureCreateInfoKHR createInfo{};
        createInfo.setBuffer(*buffer.buffer);
        createInfo.setSize(buildSizesInfo.accelerationStructureSize);
        createInfo.setType(type);
        accel = device.createAccelerationStructureKHRUnique(createInfo);

        // Create scratch buffer
        Buffer scratchBuffer;
        scratchBuffer.init(physicalDevice, device,
                           buildSizesInfo.buildScratchSize,
                           vk::BufferUsageFlagBits::eStorageBuffer |
                               vk::BufferUsageFlagBits::eShaderDeviceAddress,
                           vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Create build info
        vk::AccelerationStructureBuildGeometryInfoKHR geometryInfo{};
        geometryInfo.setType(type);
        geometryInfo.setFlags(
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        geometryInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
        geometryInfo.setDstAccelerationStructure(*accel);
        geometryInfo.setGeometries(geometry);
        geometryInfo.setScratchData(scratchBuffer.deviceAddress);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.setPrimitiveCount(primitiveCount);
        buildRangeInfo.setPrimitiveOffset(0);
        buildRangeInfo.setFirstVertex(0);
        buildRangeInfo.setTransformOffset(0);

        // Build
        vkutils::oneTimeSubmit(device, commandPool, queue,
                               [&](vk::CommandBuffer commandBuffer) {
                                   commandBuffer.buildAccelerationStructuresKHR(
                                       geometryInfo, &buildRangeInfo);
                               });

        // Get address
        vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.setAccelerationStructure(*accel);
        buffer.deviceAddress =
            device.getAccelerationStructureAddressKHR(addressInfo);
    }
};

class Application {
public:
    void run() {
        initWindow();
        initVulkan();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        device->waitIdle();

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
    AccelStruct topAccel{};

    // Pipeline
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;

    // Descriptor
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSet descriptorSet;

    // Shader binding table
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;
    uint32_t handleSize{};
    uint32_t handleAlignment{};
    uint32_t handleSizeAligned{};
    Buffer raygenShaderBindingTable{};
    Buffer missShaderBindingTable{};
    Buffer hitShaderBindingTable{};

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

        physicalDevice = vkutils::pickPhysicalDevice(  //
            *instance, *surface, deviceExtensions);

        queueFamilyIndex = vkutils::findGeneralQueueFamily(  //
            physicalDevice, *surface);

        device = vkutils::createLogicalDevice(  //
            physicalDevice, queueFamilyIndex, deviceExtensions);

        queue = device->getQueue(queueFamilyIndex, 0);

        // Create command buffers
        commandPool = vkutils::createCommandPool(*device, queueFamilyIndex);
        commandBuffer = vkutils::createCommandBuffer(*device, *commandPool);

        // Create swapchain
        // Specify images as storage images
        surfaceFormat = vkutils::chooseSurfaceFormat(physicalDevice, *surface);
        swapchain = vkutils::createSwapchain(  //
            physicalDevice, *device, *surface, queueFamilyIndex,
            vk::ImageUsageFlagBits::eStorage, surfaceFormat,  //
            WIDTH, HEIGHT);
        swapchainImages = device->getSwapchainImagesKHR(*swapchain);
        createSwapchainImageViews();

        // AS
        createBottomLevelAS();
        createTopLevelAS();

        // Pipeline, DescSet
        createDescriptorPool();
        createDescSetLayout();
        prepareShaderStages();
        createRayTracingPipeline();
        createShaderBindingTable();
        createDescriptorSets();
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

    void createBottomLevelAS() {
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
        vk::AccelerationStructureGeometryTrianglesDataKHR triangleData{};
        triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangleData.setVertexData(vertexBuffer.deviceAddress);
        triangleData.setVertexStride(sizeof(Vertex));
        triangleData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
        triangleData.setIndexType(vk::IndexType::eUint32);
        triangleData.setIndexData(indexBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometry.setGeometry({triangleData});
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // Create and build BLAS
        uint32_t primitiveCount = static_cast<uint32_t>(indices.size() / 3);
        bottomAccel.init(physicalDevice, *device, *commandPool, queue,
                         vk::AccelerationStructureTypeKHR::eBottomLevel,
                         geometry, primitiveCount);
    }

    void createTopLevelAS() {
        // Create instance
        vk::TransformMatrixKHR transformMatrix = std::array{
            std::array{1.0f, 0.0f, 0.0f, 0.0f},
            std::array{0.0f, 1.0f, 0.0f, 0.0f},
            std::array{0.0f, 0.0f, 1.0f, 0.0f},
        };

        vk::AccelerationStructureInstanceKHR accelInstance{};
        accelInstance.setTransform(transformMatrix);
        accelInstance.setInstanceCustomIndex(0);
        accelInstance.setMask(0xFF);
        accelInstance.setInstanceShaderBindingTableRecordOffset(0);
        accelInstance.setFlags(
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        accelInstance.setAccelerationStructureReference(
            bottomAccel.buffer.deviceAddress);

        Buffer instancesBuffer;
        instancesBuffer.init(
            physicalDevice, *device,
            sizeof(vk::AccelerationStructureInstanceKHR),
            vk::BufferUsageFlagBits::
                    eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            &accelInstance);

        // Create geometry
        vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
        instancesData.setArrayOfPointers(false);
        instancesData.setData(instancesBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometry.setGeometry({instancesData});
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // Create and build TLAS
        constexpr uint32_t primitiveCount = 1;
        topAccel.init(physicalDevice, *device, *commandPool, queue,
                      vk::AccelerationStructureTypeKHR::eTopLevel, geometry,
                      primitiveCount);
    }

    void createDescSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> bindings(2);
        // [0]: For AS
        bindings[0].setBinding(0);
        bindings[0].setDescriptorType(
            vk::DescriptorType::eAccelerationStructureKHR);
        bindings[0].setDescriptorCount(1);
        bindings[0].setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);
        // [1]: For storage image
        bindings[1].setBinding(1);
        bindings[1].setDescriptorType(vk::DescriptorType::eStorageImage);
        bindings[1].setDescriptorCount(1);
        bindings[1].setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

        vk::DescriptorSetLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.setBindings(bindings);
        descriptorSetLayout =
            device->createDescriptorSetLayoutUnique(layoutCreateInfo);
    }

    void prepareShaderStages() {
        constexpr uint32_t raygenIndex = 0;
        constexpr uint32_t missIndex = 1;
        constexpr uint32_t chitIndex = 2;
        shaderStages.resize(3);
        shaderModules.resize(3);
        shaderGroups.resize(3);

        // Ray generation group
        shaderModules[raygenIndex] = vkutils::createShaderModule(
            *device, SHADER_DIR + "raygen.rgen.spv");

        shaderStages[raygenIndex].setStage(vk::ShaderStageFlagBits::eRaygenKHR);
        shaderStages[raygenIndex].setModule(*shaderModules[raygenIndex]);
        shaderStages[raygenIndex].setPName("main");

        shaderGroups[raygenIndex].setType(
            vk::RayTracingShaderGroupTypeKHR::eGeneral);
        shaderGroups[raygenIndex].setGeneralShader(raygenIndex);
        shaderGroups[raygenIndex].setClosestHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[raygenIndex].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[raygenIndex].setIntersectionShader(VK_SHADER_UNUSED_KHR);

        // Ray miss group
        shaderModules[missIndex] =
            vkutils::createShaderModule(*device, SHADER_DIR + "miss.rmiss.spv");

        shaderStages[missIndex].setStage(vk::ShaderStageFlagBits::eMissKHR);
        shaderStages[missIndex].setModule(*shaderModules[missIndex]);
        shaderStages[missIndex].setPName("main");

        shaderGroups[missIndex].setType(
            vk::RayTracingShaderGroupTypeKHR::eGeneral);
        shaderGroups[missIndex].setGeneralShader(missIndex);
        shaderGroups[missIndex].setClosestHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[missIndex].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[missIndex].setIntersectionShader(VK_SHADER_UNUSED_KHR);

        // Ray closest hit group
        shaderModules[chitIndex] = vkutils::createShaderModule(
            *device, SHADER_DIR + "closesthit.rchit.spv");

        shaderStages[chitIndex].setStage(
            vk::ShaderStageFlagBits::eClosestHitKHR);
        shaderStages[chitIndex].setModule(*shaderModules[chitIndex]);
        shaderStages[chitIndex].setPName("main");

        shaderGroups[chitIndex].setType(
            vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
        shaderGroups[chitIndex].setGeneralShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[chitIndex].setClosestHitShader(chitIndex);
        shaderGroups[chitIndex].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[chitIndex].setIntersectionShader(VK_SHADER_UNUSED_KHR);
    }

    void createRayTracingPipeline() {
        // Create pipeline layout
        vk::PipelineLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.setSetLayouts(*descriptorSetLayout);
        pipelineLayout = device->createPipelineLayoutUnique(layoutCreateInfo);

        // Create pipeline
        vk::RayTracingPipelineCreateInfoKHR createInfo{};
        createInfo.setLayout(*pipelineLayout);
        createInfo.setStages(shaderStages);
        createInfo.setGroups(shaderGroups);
        createInfo.setMaxPipelineRayRecursionDepth(1);
        auto result = device->createRayTracingPipelineKHRUnique(
            nullptr, nullptr, createInfo);
        if (result.result != vk::Result::eSuccess) {
            std::cerr << "Failed to create ray tracing pipeline.\n";
            std::abort();
        }
        pipeline = std::move(result.value);
    }

    void createShaderBindingTable() {
        // Get RT props
        auto rayTracingPipelineProperties =
            vkutils::getRayTracingProps(physicalDevice);
        handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
        handleAlignment =
            rayTracingPipelineProperties.shaderGroupHandleAlignment;
        handleSizeAligned =
            vkutils::getAlignedSize(handleSize, handleAlignment);

        const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
        const uint32_t sbtSize = groupCount * handleSizeAligned;

        // Get shader group handles
        std::vector<uint8_t> shaderHandleStorage(sbtSize);
        auto result = device->getRayTracingShaderGroupHandlesKHR(
            *pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to get ray tracing shader group handles.\n";
            std::abort();
        }

        // Create SBT
        vk::BufferUsageFlags sbtBufferUsageFlags =
            vk::BufferUsageFlagBits::eShaderBindingTableKHR |
            vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eShaderDeviceAddress;
        vk::MemoryPropertyFlags sbtMemoryProperty =
            vk::MemoryPropertyFlagBits::eHostVisible |  //
            vk::MemoryPropertyFlagBits::eHostCoherent;
        raygenShaderBindingTable.init(
            physicalDevice, *device,  //
            handleSize, sbtBufferUsageFlags, sbtMemoryProperty,
            shaderHandleStorage.data() + 0 * handleSizeAligned);
        missShaderBindingTable.init(
            physicalDevice, *device,  //
            handleSize, sbtBufferUsageFlags, sbtMemoryProperty,
            shaderHandleStorage.data() + 1 * handleSizeAligned);
        hitShaderBindingTable.init(
            physicalDevice, *device,  //
            handleSize, sbtBufferUsageFlags, sbtMemoryProperty,
            shaderHandleStorage.data() + 2 * handleSizeAligned);
    }

    void createDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eAccelerationStructureKHR, 1},
            {vk::DescriptorType::eStorageImage, 1},
        };

        vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo{};
        descriptorPoolCreateInfo.setPoolSizes(poolSizes);
        descriptorPoolCreateInfo.setMaxSets(1);
        descriptorPoolCreateInfo.setFlags(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descriptorPool =
            device->createDescriptorPoolUnique(descriptorPoolCreateInfo);
    }

    void createDescriptorSets() {
        vk::DescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.setDescriptorPool(*descriptorPool);
        allocateInfo.setSetLayouts(*descriptorSetLayout);
        descriptorSet = std::move(
            device->allocateDescriptorSetsUnique(allocateInfo).front());
    }

    void updateDescriptorSets(vk::ImageView imageView) {
        std::vector<vk::WriteDescriptorSet> writes(2);

        // [0]: For AS
        vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
        accelInfo.setAccelerationStructures(*topAccel.accel);
        writes[0].setDstSet(*descriptorSet);
        writes[0].setDstBinding(0);
        writes[0].setDescriptorCount(1);
        writes[0].setDescriptorType(
            vk::DescriptorType::eAccelerationStructureKHR);
        writes[0].setPNext(&accelInfo);

        // [0]: For storage image
        vk::DescriptorImageInfo imageDescriptor{};
        imageDescriptor.setImageView(imageView);
        imageDescriptor.setImageLayout(vk::ImageLayout::eGeneral);
        writes[1].setDstSet(*descriptorSet);
        writes[1].setDstBinding(1);
        writes[1].setDescriptorType(vk::DescriptorType::eStorageImage);
        writes[1].setImageInfo(imageDescriptor);

        // Update
        device->updateDescriptorSets(writes, nullptr);
    }

    void recordCommandBuffer(vk::Image image) {
        // Begin
        commandBuffer->begin(vk::CommandBufferBeginInfo{});

        // Set image layout to general
        vk::ImageSubresourceRange subresourceRange{};
        subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
        subresourceRange.setBaseMipLevel(0);
        subresourceRange.setLevelCount(1);
        subresourceRange.setBaseArrayLayer(0);
        subresourceRange.setLayerCount(1);
        vkutils::setImageLayout(*commandBuffer, image,  //
                                vk::ImageLayout::ePresentSrcKHR,
                                vk::ImageLayout::eGeneral, subresourceRange);

        // Bind pipeline
        commandBuffer->bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                                    *pipeline);

        // Bind desc set
        commandBuffer->bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR,  // pipelineBindPoint
            *pipelineLayout,                        // layout
            0,                                      // firstSet
            *descriptorSet,                         // descriptorSets
            nullptr                                 // dynamicOffsets
        );

        // Trace rays
        vk::StridedDeviceAddressRegionKHR raygenEntry{};
        raygenEntry.setDeviceAddress(raygenShaderBindingTable.deviceAddress);
        raygenEntry.setStride(handleSizeAligned);
        raygenEntry.setSize(handleSizeAligned);

        vk::StridedDeviceAddressRegionKHR missEntry{};
        missEntry.setDeviceAddress(missShaderBindingTable.deviceAddress);
        missEntry.setStride(handleSizeAligned);
        missEntry.setSize(handleSizeAligned);

        vk::StridedDeviceAddressRegionKHR hitEntry{};
        hitEntry.setDeviceAddress(hitShaderBindingTable.deviceAddress);
        hitEntry.setStride(handleSizeAligned);
        hitEntry.setSize(handleSizeAligned);

        commandBuffer->traceRaysKHR(raygenEntry,  // raygenShaderBindingTable
                                    missEntry,    // missShaderBindingTable
                                    hitEntry,     // hitShaderBindingTable
                                    {},           // callableShaderBindingTable
                                    WIDTH,        // width
                                    HEIGHT,       // height
                                    1             // depth
        );

        // Set image layout to present src
        vkutils::setImageLayout(*commandBuffer, image,  //
                                vk::ImageLayout::eGeneral,
                                vk::ImageLayout::ePresentSrcKHR,
                                subresourceRange);

        // End
        commandBuffer->end();
    }

    void drawFrame() {
        static int frame = 0;
        std::cout << frame << '\n';

        // Create semaphore
        vk::SemaphoreCreateInfo semaphoreCreateInfo{};
        vk::UniqueSemaphore imageAvailableSemaphore =
            device->createSemaphoreUnique(semaphoreCreateInfo);

        // Acquire next image
        auto result = device->acquireNextImageKHR(
            *swapchain, std::numeric_limits<uint64_t>::max(),
            *imageAvailableSemaphore);
        if (result.result != vk::Result::eSuccess) {
            std::cerr << "Failed to acquire next image.\n";
            std::abort();
        }
        uint32_t imageIndex = result.value;

        // Update descriptor sets using current image
        updateDescriptorSets(*swapchainImageViews[imageIndex]);

        // Record command buffer
        recordCommandBuffer(swapchainImages[imageIndex]);

        // Submit command buffer
        vk::PipelineStageFlags waitStage{
            vk::PipelineStageFlagBits::eRayTracingShaderKHR};
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*commandBuffer);
        submitInfo.setWaitSemaphores(*imageAvailableSemaphore);
        queue.submit(submitInfo);

        // Wait
        queue.waitIdle();

        // Present
        vk::PresentInfoKHR presentInfo{};
        presentInfo.setSwapchains(*swapchain);
        presentInfo.setImageIndices(imageIndex);
        vk::Result presentResult = queue.presentKHR(presentInfo);
        if (presentResult != vk::Result::eSuccess) {
            std::cerr << "Failed to present.\n";
            std::abort();
        }

        frame++;
    }
};
