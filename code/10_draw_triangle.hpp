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

    // Descriptor
    vk::UniqueDescriptorPool descPool;
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descSet;

    // Pipeline
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;

    // Shader binding table
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    Buffer sbt{};
    vk::StridedDeviceAddressRegionKHR raygenRegion{};
    vk::StridedDeviceAddressRegionKHR missRegion{};
    vk::StridedDeviceAddressRegionKHR hitRegion{};

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

        // Shader
        prepareShaders();

        // Pipeline, DescSet
        createDescriptorPool();
        createDescSetLayout();
        createDescriptorSet();

        createRayTracingPipeline();
        createShaderBindingTable();
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
                    vkutils::setImageLayout(commandBuffer, image,  //
                                            vk::ImageLayout::eUndefined,
                                            vk::ImageLayout::ePresentSrcKHR);
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

    void createTopLevelAS() {
        std::cout << "Create TLAS\n";

        // Create instance
        vk::TransformMatrixKHR transform = std::array{
            std::array{1.0f, 0.0f, 0.0f, 0.0f},
            std::array{0.0f, 1.0f, 0.0f, 0.0f},
            std::array{0.0f, 0.0f, 1.0f, 0.0f},
        };

        vk::AccelerationStructureInstanceKHR accelInstance{};
        accelInstance.setTransform(transform);
        accelInstance.setInstanceCustomIndex(0);
        accelInstance.setMask(0xFF);
        accelInstance.setInstanceShaderBindingTableRecordOffset(0);
        accelInstance.setFlags(
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        accelInstance.setAccelerationStructureReference(
            bottomAccel.buffer.address);

        Buffer instanceBuffer;
        instanceBuffer.init(
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
        instancesData.setData(instanceBuffer.address);

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

    void addShader(uint32_t shaderIndex,
                   const std::string& filename,
                   vk::ShaderStageFlagBits stage) {
        shaderModules[shaderIndex] =
            vkutils::createShaderModule(*device, SHADER_DIR + filename);
        shaderStages[shaderIndex].setStage(stage);
        shaderStages[shaderIndex].setModule(*shaderModules[shaderIndex]);
        shaderStages[shaderIndex].setPName("main");
    }

    void prepareShaders() {
        // Create shader modules and shader stages
        uint32_t raygenShader = 0;
        uint32_t missShader = 1;
        uint32_t chitShader = 2;
        shaderStages.resize(3);
        shaderModules.resize(3);

        addShader(raygenShader, "raygen.rgen.spv",  //
                  vk::ShaderStageFlagBits::eRaygenKHR);
        addShader(missShader, "miss.rmiss.spv",  //
                  vk::ShaderStageFlagBits::eMissKHR);
        addShader(chitShader, "closesthit.rchit.spv",
                  vk::ShaderStageFlagBits::eClosestHitKHR);

        // Create shader groups
        uint32_t raygenGroup = 0;
        uint32_t missGroup = 1;
        uint32_t hitGroup = 2;
        shaderGroups.resize(3);

        // Raygen group
        shaderGroups[raygenGroup].setType(
            vk::RayTracingShaderGroupTypeKHR::eGeneral);
        shaderGroups[raygenGroup].setGeneralShader(raygenShader);
        shaderGroups[raygenGroup].setClosestHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[raygenGroup].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[raygenGroup].setIntersectionShader(VK_SHADER_UNUSED_KHR);

        // Miss group
        shaderGroups[missGroup].setType(
            vk::RayTracingShaderGroupTypeKHR::eGeneral);
        shaderGroups[missGroup].setGeneralShader(missShader);
        shaderGroups[missGroup].setClosestHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[missGroup].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[missGroup].setIntersectionShader(VK_SHADER_UNUSED_KHR);

        // Hit group
        shaderGroups[hitGroup].setType(
            vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
        shaderGroups[hitGroup].setGeneralShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[hitGroup].setClosestHitShader(chitShader);
        shaderGroups[hitGroup].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[hitGroup].setIntersectionShader(VK_SHADER_UNUSED_KHR);
    }

    void createDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eAccelerationStructureKHR, 1},
            {vk::DescriptorType::eStorageImage, 1},
        };

        vk::DescriptorPoolCreateInfo createInfo{};
        createInfo.setPoolSizes(poolSizes);
        createInfo.setMaxSets(1);
        createInfo.setFlags(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descPool = device->createDescriptorPoolUnique(createInfo);
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

        vk::DescriptorSetLayoutCreateInfo createInfo{};
        createInfo.setBindings(bindings);
        descSetLayout = device->createDescriptorSetLayoutUnique(createInfo);
    }

    void createDescriptorSet() {
        std::cout << "Create desc set\n";

        vk::DescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.setDescriptorPool(*descPool);
        allocateInfo.setSetLayouts(*descSetLayout);
        descSet = std::move(
            device->allocateDescriptorSetsUnique(allocateInfo).front());
    }

    void createRayTracingPipeline() {
        std::cout << "Create pipeline\n";

        // Create pipeline layout
        vk::PipelineLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.setSetLayouts(*descSetLayout);
        pipelineLayout = device->createPipelineLayoutUnique(layoutCreateInfo);

        // Create pipeline
        vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo{};
        pipelineCreateInfo.setLayout(*pipelineLayout);
        pipelineCreateInfo.setStages(shaderStages);
        pipelineCreateInfo.setGroups(shaderGroups);
        pipelineCreateInfo.setMaxPipelineRayRecursionDepth(1);
        auto result = device->createRayTracingPipelineKHRUnique(
            nullptr, nullptr, pipelineCreateInfo);
        if (result.result != vk::Result::eSuccess) {
            std::cerr << "Failed to create ray tracing pipeline.\n";
            std::abort();
        }
        pipeline = std::move(result.value);
    }

    void createShaderBindingTable() {
        // Get RT props
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties =
            vkutils::getRayTracingProps(physicalDevice);
        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;
        uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;
        uint32_t handleSizeAligned =
            vkutils::alignUp(handleSize, handleAlignment);

        // Set strides and sizes
        uint32_t raygenShaderCount = 1;  // raygen count must be 1
        uint32_t missShaderCount = 1;
        uint32_t hitShaderCount = 1;

        raygenRegion.setStride(
            vkutils::alignUp(handleSizeAligned, baseAlignment));
        raygenRegion.setSize(raygenRegion.stride);

        missRegion.setStride(handleSizeAligned);
        missRegion.setSize(vkutils::alignUp(missShaderCount * handleSizeAligned,
                                            baseAlignment));

        hitRegion.setStride(handleSizeAligned);
        hitRegion.setSize(vkutils::alignUp(hitShaderCount * handleSizeAligned,
                                           baseAlignment));

        // Create SBT
        vk::DeviceSize sbtSize =
            raygenRegion.size + missRegion.size + hitRegion.size;
        sbt.init(physicalDevice, *device, sbtSize,
                 vk::BufferUsageFlagBits::eShaderBindingTableKHR |
                     vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent);

        // Get shader group handles
        uint32_t handleCount =
            raygenShaderCount + missShaderCount + hitShaderCount;
        uint32_t handleStorageSize = handleCount * handleSize;
        std::vector<uint8_t> handleStorage(handleStorageSize);
        auto result = device->getRayTracingShaderGroupHandlesKHR(
            *pipeline, 0, handleCount, handleStorageSize, handleStorage.data());
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to get ray tracing shader group handles.\n";
            std::abort();
        }

        // Copy handles
        uint32_t handleIndex = 0;
        uint8_t* sbtHead =
            static_cast<uint8_t*>(device->mapMemory(*sbt.memory, 0, sbtSize));

        uint8_t* dstPtr = sbtHead;
        auto copyHandle = [&](uint32_t index) {
            std::memcpy(dstPtr, handleStorage.data() + handleSize * index,
                        handleSize);
        };

        // Raygen
        copyHandle(handleIndex++);

        // Miss
        dstPtr = sbtHead + raygenRegion.size;
        for (uint32_t c = 0; c < missShaderCount; c++) {
            copyHandle(handleIndex++);
            dstPtr += missRegion.stride;
        }

        // Hit
        dstPtr = sbtHead + raygenRegion.size + missRegion.size;
        for (uint32_t c = 0; c < hitShaderCount; c++) {
            copyHandle(handleIndex++);
            dstPtr += hitRegion.stride;
        }

        raygenRegion.setDeviceAddress(sbt.address);
        missRegion.setDeviceAddress(sbt.address + raygenRegion.size);
        hitRegion.setDeviceAddress(sbt.address + raygenRegion.size +
                                   missRegion.size);
    }

    void updateDescriptorSet(vk::ImageView imageView) {
        std::vector<vk::WriteDescriptorSet> writes(2);

        // [0]: For AS
        vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
        accelInfo.setAccelerationStructures(*topAccel.accel);
        writes[0].setDstSet(*descSet);
        writes[0].setDstBinding(0);
        writes[0].setDescriptorCount(1);
        writes[0].setDescriptorType(
            vk::DescriptorType::eAccelerationStructureKHR);
        writes[0].setPNext(&accelInfo);

        // [0]: For storage image
        vk::DescriptorImageInfo imageInfo{};
        imageInfo.setImageView(imageView);
        imageInfo.setImageLayout(vk::ImageLayout::eGeneral);
        writes[1].setDstSet(*descSet);
        writes[1].setDstBinding(1);
        writes[1].setDescriptorType(vk::DescriptorType::eStorageImage);
        writes[1].setImageInfo(imageInfo);

        // Update
        device->updateDescriptorSets(writes, nullptr);
    }

    void recordCommandBuffer(vk::Image image) {
        // Begin
        commandBuffer->begin(vk::CommandBufferBeginInfo{});

        // Set image layout to general
        vkutils::setImageLayout(*commandBuffer, image,  //
                                vk::ImageLayout::ePresentSrcKHR,
                                vk::ImageLayout::eGeneral);

        // Bind pipeline
        commandBuffer->bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                                    *pipeline);

        // Bind desc set
        commandBuffer->bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR,  // pipelineBindPoint
            *pipelineLayout,                        // layout
            0,                                      // firstSet
            *descSet,                               // descSets
            nullptr                                 // dynamicOffsets
        );

        // Trace rays
        commandBuffer->traceRaysKHR(  //
            raygenRegion,             // raygen
            missRegion,               // miss
            hitRegion,                // hit
            {},                       // callable
            WIDTH, HEIGHT, 1          // width, height, depth
        );

        // Set image layout to present src
        vkutils::setImageLayout(*commandBuffer, image,  //
                                vk::ImageLayout::eGeneral,
                                vk::ImageLayout::ePresentSrcKHR);

        // End
        commandBuffer->end();
    }

    void drawFrame() {
        static int frame = 0;
        std::cout << frame << '\n';

        // Create semaphore
        vk::UniqueSemaphore imageAvailableSemaphore =
            device->createSemaphoreUnique({});

        // Acquire next image
        auto result = device->acquireNextImageKHR(
            *swapchain, std::numeric_limits<uint64_t>::max(),
            *imageAvailableSemaphore);
        if (result.result != vk::Result::eSuccess) {
            std::cerr << "Failed to acquire next image.\n";
            std::abort();
        }

        // Update descriptor sets using current image
        uint32_t imageIndex = result.value;
        updateDescriptorSet(*swapchainImageViews[imageIndex]);

        // Record command buffer
        recordCommandBuffer(swapchainImages[imageIndex]);

        // Submit command buffer
        vk::PipelineStageFlags waitStage{vk::PipelineStageFlagBits::eTopOfPipe};
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
        if (queue.presentKHR(presentInfo) != vk::Result::eSuccess) {
            std::cerr << "Failed to present.\n";
            std::abort();
        }

        frame++;
    }
};
