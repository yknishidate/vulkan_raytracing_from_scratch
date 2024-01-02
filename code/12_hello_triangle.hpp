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
        Buffer scratchBuffer = createScratchBuffer(buildSizesInfo.buildScratchSize);

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
        bottomAccel.buffer = createAccelerationStructureBuffer(buildSizesInfo);

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
        topAccel.buffer = createAccelerationStructureBuffer(buildSizesInfo);

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
        std::array<vk::PipelineShaderStageCreateInfo, 3> shaderStages;
        constexpr uint32_t shaderIndexRaygen = 0;
        constexpr uint32_t shaderIndexMiss = 1;
        constexpr uint32_t shaderIndexClosestHit = 2;

        std::vector<vk::UniqueShaderModule> shaderModules;

        // Ray generation グループ
        shaderModules.push_back(
            vkutils::createShaderModule(*device, SHADER_DIR + "raygen.rgen.spv"));
        shaderStages[shaderIndexRaygen] = vk::PipelineShaderStageCreateInfo{}
                                              .setStage(vk::ShaderStageFlagBits::eRaygenKHR)
                                              .setModule(*shaderModules.back())
                                              .setPName("main");
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{}
                                   .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
                                   .setGeneralShader(shaderIndexRaygen)
                                   .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                                   .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                                   .setIntersectionShader(VK_SHADER_UNUSED_KHR));

        // Ray miss グループ
        shaderModules.push_back(
            vkutils::createShaderModule(*device, SHADER_DIR + "miss.rmiss.spv"));
        shaderStages[shaderIndexMiss] = vk::PipelineShaderStageCreateInfo{}
                                            .setStage(vk::ShaderStageFlagBits::eMissKHR)
                                            .setModule(*shaderModules.back())
                                            .setPName("main");
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{}
                                   .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
                                   .setGeneralShader(shaderIndexMiss)
                                   .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                                   .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                                   .setIntersectionShader(VK_SHADER_UNUSED_KHR));

        // Ray closest hit グループ
        shaderModules.push_back(
            vkutils::createShaderModule(*device, SHADER_DIR + "closesthit.rchit.spv"));
        shaderStages[shaderIndexClosestHit] = vk::PipelineShaderStageCreateInfo{}
                                                  .setStage(vk::ShaderStageFlagBits::eClosestHitKHR)
                                                  .setModule(*shaderModules.back())
                                                  .setPName("main");
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{}
                                   .setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup)
                                   .setGeneralShader(VK_SHADER_UNUSED_KHR)
                                   .setClosestHitShader(shaderIndexClosestHit)
                                   .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                                   .setIntersectionShader(VK_SHADER_UNUSED_KHR));

        // レイトレーシングパイプラインを作成する
        auto result =
            device->createRayTracingPipelineKHRUnique(nullptr, nullptr,
                                                      vk::RayTracingPipelineCreateInfoKHR{}
                                                          .setStages(shaderStages)
                                                          .setGroups(shaderGroups)
                                                          .setMaxPipelineRayRecursionDepth(1)
                                                          .setLayout(*pipelineLayout));
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
            {vk::DescriptorType::eStorageImage, 1}};

        vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo{};
        descriptorPoolCreateInfo.setPoolSizes(poolSizes);
        descriptorPoolCreateInfo.setMaxSets(1);
        descriptorPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descriptorPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);

        // ディスクリプタセットを1つ準備する
        auto descriptorSets =
            device->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo{}
                                                     .setDescriptorPool(*descriptorPool)
                                                     .setSetLayouts(*descriptorSetLayout));
        descriptorSet = std::move(descriptorSets.front());
    }

    void updateDescriptorSets(vk::ImageView imageView) {
        // Top Level ASをシェーダにバインドするためのディスクリプタ
        vk::WriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
        descriptorAccelerationStructureInfo.setAccelerationStructures(*topAccel.handle);

        vk::WriteDescriptorSet accelerationStructureWrite{};
        accelerationStructureWrite.setDstSet(*descriptorSet)
            .setDstBinding(0)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
            .setPNext(&descriptorAccelerationStructureInfo);

        // Storage imageのためのディスクリプタ
        vk::DescriptorImageInfo imageDescriptor{};
        imageDescriptor.setImageView(imageView).setImageLayout(vk::ImageLayout::eGeneral);

        vk::WriteDescriptorSet resultImageWrite{};
        resultImageWrite.setDstSet(*descriptorSet)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setDstBinding(1)
            .setImageInfo(imageDescriptor);

        device->updateDescriptorSets({accelerationStructureWrite, resultImageWrite}, nullptr);
    }

    void buildCommandBuffers(vk::CommandBuffer commandBuffer, vk::Image image) {
        vk::ImageSubresourceRange subresourceRange{};
        subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1);

        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        vkutils::setImageLayout(commandBuffer, image, vk::ImageLayout::ePresentSrcKHR,
                                vk::ImageLayout::eGeneral, subresourceRange);

        const uint32_t handleSizeAligned =
            vkutils::getHandleSizeAligned(rayTracingPipelineProperties);

        vk::StridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
        raygenShaderSbtEntry.setDeviceAddress(raygenShaderBindingTable.deviceAddress)
            .setStride(handleSizeAligned)
            .setSize(handleSizeAligned);

        vk::StridedDeviceAddressRegionKHR missShaderSbtEntry{};
        missShaderSbtEntry.setDeviceAddress(missShaderBindingTable.deviceAddress)
            .setStride(handleSizeAligned)
            .setSize(handleSizeAligned);

        vk::StridedDeviceAddressRegionKHR hitShaderSbtEntry{};
        hitShaderSbtEntry.setDeviceAddress(hitShaderBindingTable.deviceAddress)
            .setStride(handleSizeAligned)
            .setSize(handleSizeAligned);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR,  // pipelineBindPoint
            *pipelineLayout,                        // layout
            0,                                      // firstSet
            *descriptorSet,                         // descriptorSets
            nullptr                                 // dynamicOffsets
        );

        // レイトレーシングを実行
        commandBuffer.traceRaysKHR(raygenShaderSbtEntry,  // raygenShaderBindingTable
                                   missShaderSbtEntry,    // missShaderBindingTable
                                   hitShaderSbtEntry,     // hitShaderBindingTable
                                   {},                    // callableShaderBindingTable
                                   WIDTH,                 // width
                                   HEIGHT,                // height
                                   1                      // depth
        );

        // スワップチェインの画像を提示用に設定
        vkutils::setImageLayout(commandBuffer, image, vk::ImageLayout::eGeneral,
                                vk::ImageLayout::ePresentSrcKHR, subresourceRange);
        commandBuffer.end();
    }

    void drawFrame() {
        static int frame = 0;
        std::cout << frame << '\n';

        // 次に表示する画像のインデックスをスワップチェインから取得する
        vk::SemaphoreCreateInfo semaphoreCreateInfo{};
        vk::UniqueSemaphore imageAvailableSemaphore =
            device->createSemaphoreUnique(semaphoreCreateInfo);
        auto result = device->acquireNextImageKHR(*swapchain,  // swapchain
                                                  std::numeric_limits<uint64_t>::max(),  // timeout
                                                  *imageAvailableSemaphore);
        if (result.result != vk::Result::eSuccess) {
            std::cerr << "Failed to acquire next image.\n";
            std::abort();
        }

        uint32_t imageIndex = result.value;

        updateDescriptorSets(*swapchainImageViews[imageIndex]);

        buildCommandBuffers(*commandBuffers[imageIndex], swapchainImages[imageIndex]);

        // レイトレーシングを行うコマンドバッファを実行する
        vk::PipelineStageFlags waitStage{vk::PipelineStageFlagBits::eRayTracingShaderKHR};
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*commandBuffers[imageIndex]);
        submitInfo.setWaitSemaphores(*imageAvailableSemaphore);
        queue.submit(submitInfo);
        queue.waitIdle();

        // 表示する
        vk::PresentInfoKHR presentInfo{};
        presentInfo.setSwapchains(*swapchain);
        presentInfo.setImageIndices(imageIndex);
        queue.presentKHR(presentInfo);

        frame++;
    }

    Buffer createBuffer(vk::DeviceSize size,
                        vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags memoryProperty,
                        const void* data = nullptr) {
        // Bufferオブジェクトを作成
        Buffer buffer{};
        buffer.handle = device->createBufferUnique(
            vk::BufferCreateInfo{}.setSize(size).setUsage(usage).setQueueFamilyIndexCount(0));

        // メモリを確保してバインドする
        auto memoryRequirements = device->getBufferMemoryRequirements(*buffer.handle);
        vk::MemoryAllocateFlagsInfo memoryFlagsInfo{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            memoryFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        }

        buffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
                .setAllocationSize(memoryRequirements.size)
                .setMemoryTypeIndex(
                    vkutils::getMemoryType(physicalDevice, memoryRequirements, memoryProperty))
                .setPNext(&memoryFlagsInfo));
        device->bindBufferMemory(*buffer.handle, *buffer.deviceMemory, 0);

        // データをメモリにコピーする
        if (data) {
            void* dataPtr = device->mapMemory(*buffer.deviceMemory, 0, size);
            memcpy(dataPtr, data, size);
            device->unmapMemory(*buffer.deviceMemory);
        }

        // バッファのデバイスアドレスを取得する
        buffer.deviceAddress = getBufferDeviceAddress(*buffer.handle);

        return buffer;
    }

    uint64_t getBufferDeviceAddress(vk::Buffer buffer) {
        vk::BufferDeviceAddressInfoKHR bufferDeviceAI{buffer};
        return device->getBufferAddressKHR(&bufferDeviceAI);
    }

    Buffer createAccelerationStructureBuffer(
        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo) {
        // Bufferオブジェクトを作成
        Buffer buffer{};
        buffer.handle = device->createBufferUnique(
            vk::BufferCreateInfo{}
                .setSize(buildSizesInfo.accelerationStructureSize)
                .setUsage(vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                          vk::BufferUsageFlagBits::eShaderDeviceAddress));

        // メモリを確保してバインドする
        auto memoryRequirements = device->getBufferMemoryRequirements(*buffer.handle);
        vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{
            vk::MemoryAllocateFlagBits::eDeviceAddress};

        buffer.deviceMemory =
            device->allocateMemoryUnique(vk::MemoryAllocateInfo{}
                                             .setAllocationSize(memoryRequirements.size)
                                             .setMemoryTypeIndex(vkutils::getMemoryType(
                                                 physicalDevice, memoryRequirements,
                                                 vk::MemoryPropertyFlagBits::eHostVisible |
                                                     vk::MemoryPropertyFlagBits::eHostCoherent))
                                             .setPNext(&memoryAllocateFlagsInfo));
        device->bindBufferMemory(*buffer.handle, *buffer.deviceMemory, 0);

        return buffer;
    }

    Buffer createScratchBuffer(vk::DeviceSize size) {
        Buffer scratchBuffer;

        // バッファを作成する
        scratchBuffer.handle =
            device->createBufferUnique(vk::BufferCreateInfo{}.setSize(size).setUsage(
                vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eShaderDeviceAddress));

        // メモリを確保してバインドする
        auto memoryRequirements = device->getBufferMemoryRequirements(*scratchBuffer.handle);
        vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{
            vk::MemoryAllocateFlagBits::eDeviceAddress};

        scratchBuffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
                .setAllocationSize(memoryRequirements.size)
                .setMemoryTypeIndex(vkutils::getMemoryType(
                    physicalDevice, memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal))
                .setPNext(&memoryAllocateFlagsInfo));
        device->bindBufferMemory(*scratchBuffer.handle, *scratchBuffer.deviceMemory, 0);

        // バッファのデバイスアドレスを取得する
        scratchBuffer.deviceAddress = getBufferDeviceAddress(*scratchBuffer.handle);

        return scratchBuffer;
    }
};
