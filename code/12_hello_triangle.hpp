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
    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{};
    vk::UniqueDevice device;
    uint32_t queueFamilyIndex{};
    vk::Queue queue;

    // Swapchain
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueImageView> swapchainImageViews;

    // Command buffer
    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;

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
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;
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
            // Swapchain
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            // レイトレーシング
            VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        };

        instance = vkutils::createInstance(layers);
        debugUtilsMessenger = vkutils::createDebugMessenger(*instance);
        surface = vkutils::createSurface(*instance, window);
        physicalDevice = vkutils::pickPhysicalDevice(*instance, *surface, deviceExtensions);
        queueFamilyIndex = vkutils::findGeneralQueueFamilies(physicalDevice, *surface);
        device = vkutils::createLogicalDevice(physicalDevice, queueFamilyIndex, deviceExtensions);
        queue = device->getQueue(queueFamilyIndex, 0);

        rayTracingPipelineProperties = vkutils::getRayTracingProps(physicalDevice);

        swapchain = vkutils::createSwapchain(physicalDevice, *device, *surface, queueFamilyIndex,
                                             WIDTH, HEIGHT);
        swapchainImages = device->getSwapchainImagesKHR(*swapchain);

        commandPool = vkutils::createCommandPool(*device, queueFamilyIndex);
        commandBuffers =
            vkutils::createDrawCommandBuffers(*device, *commandPool, swapchainImages.size());

        createSwapchainImageViews();
        createBottomLevelAS();
        createTopLevelAS();

        createDescSetLayout();
        createRayTracingPipeline();
        createShaderBindingTable();
        createDescriptorSets();
    }

    void createSwapchainImageViews() {
        for (auto& image : swapchainImages) {
            vk::ImageViewCreateInfo imageViewCreateInfo{};
            imageViewCreateInfo.setImage(image);
            imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
            imageViewCreateInfo.setFormat(vk::Format::eB8G8R8A8Unorm);
            imageViewCreateInfo.setComponents({vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                                               vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
            imageViewCreateInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            swapchainImageViews.push_back(device->createImageViewUnique(imageViewCreateInfo));
        }

        vkutils::oneTimeSubmit(*device, *commandPool, queue, [&](vk::CommandBuffer commandBuffer) {
            for (auto& image : swapchainImages) {
                vkutils::setImageLayout(commandBuffer, image, vk::ImageLayout::eUndefined,
                                        vk::ImageLayout::ePresentSrcKHR,
                                        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            }
        });
    }

    void buildAccelerationStructure(AccelStruct& accelStruct,
                                    vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo,
                                    vk::AccelerationStructureGeometryKHR geometry,
                                    vk::AccelerationStructureTypeKHR type) {
        vk::AccelerationStructureKHR handle = *accelStruct.handle;

        // スクラッチバッファを作成する
        Buffer scratchBuffer = createBuffer(
            buildSizesInfo.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        // ビルド情報を作成する
        vk::AccelerationStructureBuildGeometryInfoKHR geometryInfo{};
        geometryInfo.setType(type);
        geometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        geometryInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
        geometryInfo.setDstAccelerationStructure(handle);
        geometryInfo.setGeometries(geometry);
        geometryInfo.setScratchData(scratchBuffer.deviceAddress);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.setPrimitiveCount(1);
        buildRangeInfo.setPrimitiveOffset(0);
        buildRangeInfo.setFirstVertex(0);
        buildRangeInfo.setTransformOffset(0);

        // ビルドコマンドを送信してデバイス上でASをビルドする
        vkutils::oneTimeSubmit(*device, *commandPool, queue, [&](vk::CommandBuffer commandBuffer) {
            commandBuffer.buildAccelerationStructuresKHR(geometryInfo, &buildRangeInfo);
        });

        // アドレスを取得する
        accelStruct.buffer.deviceAddress = device->getAccelerationStructureAddressKHR({handle});
    }

    void createBottomLevelAS() {
        // 三角形のデータを用意
        std::vector<Vertex> vertices = {
            {{1.0f, 1.0f, 0.0f}},
            {{-1.0f, 1.0f, 0.0f}},
            {{0.0f, -1.0f, 0.0f}},
        };
        std::vector<uint32_t> indices = {0, 1, 2};

        // データからバッファを作成
        auto vertexBufferSize = vertices.size() * sizeof(Vertex);
        auto indexBufferSize = indices.size() * sizeof(uint32_t);
        vk::BufferUsageFlags bufferUsage{
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eStorageBuffer};
        vk::MemoryPropertyFlags memoryProperty{vk::MemoryPropertyFlagBits::eHostVisible |
                                               vk::MemoryPropertyFlagBits::eHostCoherent};
        Buffer vertexBuffer =
            createBuffer(vertexBufferSize, bufferUsage, memoryProperty, vertices.data());
        Buffer indexBuffer =
            createBuffer(indexBufferSize, bufferUsage, memoryProperty, indices.data());

        // ジオメトリには三角形データを渡す
        vk::AccelerationStructureGeometryTrianglesDataKHR triangleData{};
        triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangleData.setVertexData(vertexBuffer.deviceAddress);
        triangleData.setVertexStride(sizeof(Vertex));
        triangleData.setMaxVertex(vertices.size());
        triangleData.setIndexType(vk::IndexType::eUint32);
        triangleData.setIndexData(indexBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometry.setGeometry({triangleData});
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // ASビルドに必要なサイズを取得する
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildGeometryInfo.setGeometries(geometry);

        constexpr uint32_t primitiveCount = 1;
        auto buildSizesInfo = device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);

        // ASを保持するためのバッファを作成する
        bottomAccel.buffer = createBuffer(
            buildSizesInfo.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        // ASを作成する
        vk::AccelerationStructureCreateInfoKHR createInfo{};
        createInfo.setBuffer(*bottomAccel.buffer.handle);
        createInfo.setSize(buildSizesInfo.accelerationStructureSize);
        createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        bottomAccel.handle = device->createAccelerationStructureKHRUnique(createInfo);

        // Build
        buildAccelerationStructure(bottomAccel, buildSizesInfo, geometry,
                                   vk::AccelerationStructureTypeKHR::eBottomLevel);
    }

    void createTopLevelAS() {
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
        accelInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        accelInstance.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress);

        Buffer instancesBuffer = createBuffer(
            sizeof(vk::AccelerationStructureInstanceKHR),
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            &accelInstance);

        // Bottom Level ASを入力としてセットする
        vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
        instancesData.setArrayOfPointers(false);
        instancesData.setData(instancesBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometry.setGeometry({instancesData});
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // ASビルドに必要なサイズを取得する
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildGeometryInfo.setGeometries(geometry);

        constexpr uint32_t primitiveCount = 1;
        auto buildSizesInfo = device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);

        // ASを保持するためのバッファを作成する
        topAccel.buffer = createBuffer(
            buildSizesInfo.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        // ASを作成する
        vk::AccelerationStructureCreateInfoKHR createInfo{};
        createInfo.setBuffer(*topAccel.buffer.handle);
        createInfo.setSize(buildSizesInfo.accelerationStructureSize);
        createInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        topAccel.handle = device->createAccelerationStructureKHRUnique(createInfo);

        // Build
        buildAccelerationStructure(topAccel, buildSizesInfo, geometry,
                                   vk::AccelerationStructureTypeKHR::eTopLevel);
    }

    void createDescSetLayout() {
        // ディスクリプタセットレイアウトを作成する
        std::vector<vk::DescriptorSetLayoutBinding> bindings(2);
        bindings[0].setBinding(0);
        bindings[0].setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
        bindings[0].setDescriptorCount(1);
        bindings[0].setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);
        bindings[1].setBinding(1);
        bindings[1].setDescriptorType(vk::DescriptorType::eStorageImage);
        bindings[1].setDescriptorCount(1);
        bindings[1].setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

        vk::DescriptorSetLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.setBindings(bindings);
        descriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutCreateInfo);
    }

    void createRayTracingPipeline() {
        // パイプラインレイアウトを作成する
        vk::PipelineLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.setSetLayouts(*descriptorSetLayout);
        pipelineLayout = device->createPipelineLayoutUnique(layoutCreateInfo);

        // レイトレーシングシェーダグループの設定
        // 各シェーダグループはパイプライン内の対応するシェーダを指す
        constexpr uint32_t raygenIndex = 0;
        constexpr uint32_t missIndex = 1;
        constexpr uint32_t chitIndex = 2;
        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages(3);
        std::vector<vk::UniqueShaderModule> shaderModules(3);
        shaderGroups.resize(3);

        // Ray generation グループ
        shaderModules[raygenIndex] =
            vkutils::createShaderModule(*device, SHADER_DIR + "raygen.rgen.spv");

        shaderStages[raygenIndex].setStage(vk::ShaderStageFlagBits::eRaygenKHR);
        shaderStages[raygenIndex].setModule(*shaderModules[raygenIndex]);
        shaderStages[raygenIndex].setPName("main");

        shaderGroups[raygenIndex].setType(vk::RayTracingShaderGroupTypeKHR::eGeneral);
        shaderGroups[raygenIndex].setGeneralShader(raygenIndex);
        shaderGroups[raygenIndex].setClosestHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[raygenIndex].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[raygenIndex].setIntersectionShader(VK_SHADER_UNUSED_KHR);

        // Ray miss グループ
        shaderModules[missIndex] =
            vkutils::createShaderModule(*device, SHADER_DIR + "miss.rmiss.spv");

        shaderStages[missIndex].setStage(vk::ShaderStageFlagBits::eMissKHR);
        shaderStages[missIndex].setModule(*shaderModules[missIndex]);
        shaderStages[missIndex].setPName("main");

        shaderGroups[missIndex].setType(vk::RayTracingShaderGroupTypeKHR::eGeneral);
        shaderGroups[missIndex].setGeneralShader(missIndex);
        shaderGroups[missIndex].setClosestHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[missIndex].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[missIndex].setIntersectionShader(VK_SHADER_UNUSED_KHR);

        // Ray closest hit グループ
        shaderModules[chitIndex] =
            vkutils::createShaderModule(*device, SHADER_DIR + "closesthit.rchit.spv");

        shaderStages[chitIndex].setStage(vk::ShaderStageFlagBits::eClosestHitKHR);
        shaderStages[chitIndex].setModule(*shaderModules[chitIndex]);
        shaderStages[chitIndex].setPName("main");

        shaderGroups[chitIndex].setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
        shaderGroups[chitIndex].setGeneralShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[chitIndex].setClosestHitShader(chitIndex);
        shaderGroups[chitIndex].setAnyHitShader(VK_SHADER_UNUSED_KHR);
        shaderGroups[chitIndex].setIntersectionShader(VK_SHADER_UNUSED_KHR);

        // レイトレーシングパイプラインを作成する
        vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo{};
        pipelineCreateInfo.setStages(shaderStages);
        pipelineCreateInfo.setGroups(shaderGroups);
        pipelineCreateInfo.setMaxPipelineRayRecursionDepth(1);
        pipelineCreateInfo.setLayout(*pipelineLayout);
        auto result =
            device->createRayTracingPipelineKHRUnique(nullptr, nullptr, pipelineCreateInfo);
        if (result.result != vk::Result::eSuccess) {
            std::cerr << "Failed to create ray tracing pipeline.\n";
            std::abort();
        }

        pipeline = std::move(result.value);
    }

    void createShaderBindingTable() {
        const uint32_t handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
        const uint32_t handleSizeAligned =
            vkutils::getHandleSizeAligned(rayTracingPipelineProperties);
        const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
        const uint32_t sbtSize = groupCount * handleSizeAligned;

        const vk::BufferUsageFlags sbtBufferUsageFlags =
            vk::BufferUsageFlagBits::eShaderBindingTableKHR |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress;

        const vk::MemoryPropertyFlags sbtMemoryProperty =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        // シェーダグループのハンドルを取得する
        std::vector<uint8_t> shaderHandleStorage(sbtSize);
        auto result = device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize,
                                                                 shaderHandleStorage.data());
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to get ray tracing shader group handles.\n";
            std::abort();
        }

        // シェーダタイプごとにバインディングテーブルバッファを作成する
        raygenShaderBindingTable = createBuffer(handleSize, sbtBufferUsageFlags, sbtMemoryProperty,
                                                shaderHandleStorage.data() + 0 * handleSizeAligned);
        missShaderBindingTable = createBuffer(handleSize, sbtBufferUsageFlags, sbtMemoryProperty,
                                              shaderHandleStorage.data() + 1 * handleSizeAligned);
        hitShaderBindingTable = createBuffer(handleSize, sbtBufferUsageFlags, sbtMemoryProperty,
                                             shaderHandleStorage.data() + 2 * handleSizeAligned);
    }

    void createDescriptorSets() {
        // まずはディスクリプタプールを用意する
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eAccelerationStructureKHR, 1},
            {vk::DescriptorType::eStorageImage, 1},
        };

        vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo{};
        descriptorPoolCreateInfo.setPoolSizes(poolSizes);
        descriptorPoolCreateInfo.setMaxSets(1);
        descriptorPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descriptorPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);

        // ディスクリプタセットを1つ準備する
        vk::DescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.setDescriptorPool(*descriptorPool);
        allocateInfo.setSetLayouts(*descriptorSetLayout);
        descriptorSet = std::move(device->allocateDescriptorSetsUnique(allocateInfo).front());
    }

    void updateDescriptorSets(vk::ImageView imageView) {
        std::vector<vk::WriteDescriptorSet> writes(2);

        // Top Level ASをシェーダにバインドするためのディスクリプタ
        vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
        accelInfo.setAccelerationStructures(*topAccel.handle);

        writes[0].setDstSet(*descriptorSet);
        writes[0].setDstBinding(0);
        writes[0].setDescriptorCount(1);
        writes[0].setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
        writes[0].setPNext(&accelInfo);

        // Storage imageのためのディスクリプタ
        vk::DescriptorImageInfo imageDescriptor{};
        imageDescriptor.setImageView(imageView);
        imageDescriptor.setImageLayout(vk::ImageLayout::eGeneral);

        writes[1].setDstSet(*descriptorSet);
        writes[1].setDstBinding(1);
        writes[1].setDescriptorType(vk::DescriptorType::eStorageImage);
        writes[1].setImageInfo(imageDescriptor);

        device->updateDescriptorSets(writes, nullptr);
    }

    void recordCommandBuffer(vk::CommandBuffer commandBuffer, vk::Image image) {
        vk::ImageSubresourceRange subresourceRange{};
        subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
        subresourceRange.setBaseMipLevel(0);
        subresourceRange.setLevelCount(1);
        subresourceRange.setBaseArrayLayer(0);
        subresourceRange.setLayerCount(1);

        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        vkutils::setImageLayout(commandBuffer, image, vk::ImageLayout::ePresentSrcKHR,
                                vk::ImageLayout::eGeneral, subresourceRange);

        const uint32_t handleSizeAligned =
            vkutils::getHandleSizeAligned(rayTracingPipelineProperties);

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

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR,  // pipelineBindPoint
            *pipelineLayout,                        // layout
            0,                                      // firstSet
            *descriptorSet,                         // descriptorSets
            nullptr                                 // dynamicOffsets
        );

        // レイトレーシングを実行
        commandBuffer.traceRaysKHR(raygenEntry,  // raygenShaderBindingTable
                                   missEntry,    // missShaderBindingTable
                                   hitEntry,     // hitShaderBindingTable
                                   {},           // callableShaderBindingTable
                                   WIDTH,        // width
                                   HEIGHT,       // height
                                   1             // depth
        );

        // スワップチェインの画像を提示用に設定
        vkutils::setImageLayout(commandBuffer, image, vk::ImageLayout::eGeneral,
                                vk::ImageLayout::ePresentSrcKHR, subresourceRange);
        commandBuffer.end();
    }

    void drawFrame() {
        static int frame = 0;
        std::cout << frame << '\n';

        // Create semaphore
        vk::SemaphoreCreateInfo semaphoreCreateInfo{};
        vk::UniqueSemaphore imageAvailableSemaphore =
            device->createSemaphoreUnique(semaphoreCreateInfo);

        // Acquire next image
        auto result = device->acquireNextImageKHR(*swapchain, std::numeric_limits<uint64_t>::max(),
                                                  *imageAvailableSemaphore);
        if (result.result != vk::Result::eSuccess) {
            std::cerr << "Failed to acquire next image.\n";
            std::abort();
        }
        uint32_t imageIndex = result.value;

        // Update descriptor sets using current image
        updateDescriptorSets(*swapchainImageViews[imageIndex]);

        recordCommandBuffer(*commandBuffers[imageIndex], swapchainImages[imageIndex]);

        // Submit command buffer
        vk::PipelineStageFlags waitStage{vk::PipelineStageFlagBits::eRayTracingShaderKHR};
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*commandBuffers[imageIndex]);
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

    Buffer createBuffer(vk::DeviceSize size,
                        vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags memoryProperty,
                        const void* data = nullptr) {
        // Create buffer
        vk::BufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.setSize(size);
        bufferCreateInfo.setUsage(usage);
        bufferCreateInfo.setQueueFamilyIndexCount(0);
        vk::UniqueBuffer buffer = device->createBufferUnique(bufferCreateInfo);

        // Allocate memory
        auto memoryRequirements = device->getBufferMemoryRequirements(*buffer);
        vk::MemoryAllocateFlagsInfo memoryFlagsInfo{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            memoryFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        }

        vk::MemoryAllocateInfo allocateInfo{};
        allocateInfo.setAllocationSize(memoryRequirements.size);
        allocateInfo.setMemoryTypeIndex(
            vkutils::getMemoryType(physicalDevice, memoryRequirements, memoryProperty));
        allocateInfo.setPNext(&memoryFlagsInfo);
        vk::UniqueDeviceMemory deviceMemory = device->allocateMemoryUnique(allocateInfo);

        // Bind buffer to memory
        device->bindBufferMemory(*buffer, *deviceMemory, 0);

        // Copy data
        if (data) {
            void* dataPtr = device->mapMemory(*deviceMemory, 0, size);
            memcpy(dataPtr, data, size);
            device->unmapMemory(*deviceMemory);
        }

        // Get address
        vk::DeviceAddress deviceAddress{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            vk::BufferDeviceAddressInfoKHR bufferDeviceAI{*buffer};
            deviceAddress = device->getBufferAddressKHR(&bufferDeviceAI);
        }

        return {std::move(buffer), std::move(deviceMemory), deviceAddress};
    }
};
