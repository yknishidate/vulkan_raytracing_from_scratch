#pragma once
#include "vkutils.hpp"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

struct StorageImage {
    vk::UniqueDeviceMemory memory;
    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::Format format;
    uint32_t width;
    uint32_t height;
};

struct Vertex {
    float pos[3];
};

struct Buffer {
    vk::UniqueBuffer handle;
    vk::UniqueDeviceMemory deviceMemory;
    uint64_t deviceAddress;
};

struct AccelerationStructure {
    vk::UniqueAccelerationStructureKHR handle;
    Buffer buffer;
};

class Application {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueDevice device;
    vk::Queue graphicsQueue;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueImageView> swapchainImageViews;

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> drawCommandBuffers;

    AccelerationStructure blas{};
    AccelerationStructure tlas{};

    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;

    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;
    Buffer raygenShaderBindingTable{};
    Buffer missShaderBindingTable{};
    Buffer hitShaderBindingTable{};

    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSet descriptorSet;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        std::vector<const char*> deviceExtensions = {
            // レイトレーシング拡張
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        };
        vkutils::addDeviceExtensions(deviceExtensions);

        vkutils::enableDebugMessage();

        instance = vkutils::createInstance();
        debugUtilsMessenger = vkutils::createDebugMessenger(instance.get());
        surface = vkutils::createSurface(instance.get(), window);
        device = vkutils::createLogicalDevice(instance.get(), surface.get());
        graphicsQueue = vkutils::getGraphicsQueue(device.get());

        swapchain = vkutils::createSwapChain(device.get(), surface.get(), WIDTH, HEIGHT);
        swapchainImages = vkutils::getSwapChainImages(device.get(), swapchain.get());

        commandPool = vkutils::createCommandPool(device.get());
        drawCommandBuffers = vkutils::createDrawCommandBuffers(device.get(), commandPool.get());

        createSwapchainImageViews();
        createBottomLevelAS();
        createTopLevelAS();

        createRayTracingPipeLine();
        createShaderBindingTable();
        createDescriptorSets();
    }

    void createSwapchainImageViews() {
        for (auto& image : swapchainImages) {
            swapchainImageViews.push_back(device->createImageViewUnique(
                vk::ImageViewCreateInfo()
                    .setImage(image)
                    .setViewType(vk::ImageViewType::e2D)
                    .setFormat(vk::Format::eB8G8R8A8Unorm)
                    .setComponents({vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                                    vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA})
                    .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})));
        }

        auto commandBuffer = vkutils::createCommandBuffer(device.get(), commandPool.get(), true);
        for (auto& image : swapchainImages) {
            vkutils::setImageLayout(commandBuffer.get(), image, vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::ePresentSrcKHR,
                                    {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        }
        vkutils::submitCommandBuffer(device.get(), commandBuffer.get(), graphicsQueue);
    }

    void createBottomLevelAS() {
        // 三角形のデータを用意
        std::vector<Vertex> vertices = {
            {{1.0f, 1.0f, 0.0f}}, {{-1.0f, 1.0f, 0.0f}}, {{0.0f, -1.0f, 0.0f}}};
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
        triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat)
            .setVertexData(vertexBuffer.deviceAddress)
            .setVertexStride(sizeof(Vertex))
            .setMaxVertex(vertices.size())
            .setIndexType(vk::IndexType::eUint32)
            .setIndexData(indexBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles)
            .setGeometry({triangleData})
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // ASビルドに必要なサイズを取得する
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setGeometries(geometry);

        constexpr uint32_t primitiveCount = 1;
        auto buildSizesInfo = device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);

        // ASを保持するためのバッファを作成する
        blas.buffer = createAccelerationStructureBuffer(buildSizesInfo);

        // ASを作成する
        blas.handle = device->createAccelerationStructureKHRUnique(
            vk::AccelerationStructureCreateInfoKHR{}
                .setBuffer(blas.buffer.handle.get())
                .setSize(buildSizesInfo.accelerationStructureSize)
                .setType(vk::AccelerationStructureTypeKHR::eBottomLevel));

        // ここから実際のビルドを行っていく

        // スクラッチバッファを作成する
        Buffer scratchBuffer = createScratchBuffer(buildSizesInfo.buildScratchSize);

        // ビルド情報を作成する
        vk::AccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{};
        accelerationBuildGeometryInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
            .setDstAccelerationStructure(blas.handle.get())
            .setGeometries(geometry)
            .setScratchData(scratchBuffer.deviceAddress);

        vk::AccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
        accelerationStructureBuildRangeInfo.setPrimitiveCount(1)
            .setPrimitiveOffset(0)
            .setFirstVertex(0)
            .setTransformOffset(0);

        // ビルドコマンドを送信してデバイス上でASをビルドする
        auto commandBuffer = vkutils::createCommandBuffer(device.get(), commandPool.get(), true);
        commandBuffer->buildAccelerationStructuresKHR(accelerationBuildGeometryInfo,
                                                      &accelerationStructureBuildRangeInfo);
        vkutils::submitCommandBuffer(device.get(), commandBuffer.get(), graphicsQueue);

        // Bottom Level AS のハンドルを取得する
        blas.buffer.deviceAddress = device->getAccelerationStructureAddressKHR({blas.handle.get()});
    }

    void createTopLevelAS() {
        vk::TransformMatrixKHR transformMatrix = std::array{
            std::array{1.0f, 0.0f, 0.0f, 0.0f},
            std::array{0.0f, 1.0f, 0.0f, 0.0f},
            std::array{0.0f, 0.0f, 1.0f, 0.0f},
        };

        vk::AccelerationStructureInstanceKHR accelerationStructureInstance{};
        accelerationStructureInstance.setTransform(transformMatrix)
            .setInstanceCustomIndex(0)
            .setMask(0xFF)
            .setInstanceShaderBindingTableRecordOffset(0)
            .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable)
            .setAccelerationStructureReference(blas.buffer.deviceAddress);

        Buffer instancesBuffer = createBuffer(
            sizeof(vk::AccelerationStructureInstanceKHR),
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            &accelerationStructureInstance);

        // Bottom Level ASを入力としてセットする
        vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
        instancesData.setArrayOfPointers(false).setData(instancesBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eInstances)
            .setGeometry({instancesData})
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // ASビルドに必要なサイズを取得する
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setGeometries(geometry);

        constexpr uint32_t primitiveCount = 1;
        auto buildSizesInfo = device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);

        // ASを保持するためのバッファを作成する
        tlas.buffer = createAccelerationStructureBuffer(buildSizesInfo);

        // ASを作成する
        tlas.handle = device->createAccelerationStructureKHRUnique(
            vk::AccelerationStructureCreateInfoKHR{}
                .setBuffer(tlas.buffer.handle.get())
                .setSize(buildSizesInfo.accelerationStructureSize)
                .setType(vk::AccelerationStructureTypeKHR::eTopLevel));

        // ここから実際のビルドを行っていく

        // スクラッチバッファを作成する
        Buffer scratchBuffer = createScratchBuffer(buildSizesInfo.buildScratchSize);

        // ビルド情報を作成する
        vk::AccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{};
        accelerationBuildGeometryInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setDstAccelerationStructure(tlas.handle.get())
            .setGeometries(geometry)
            .setScratchData(scratchBuffer.deviceAddress);

        vk::AccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
        accelerationStructureBuildRangeInfo.setPrimitiveCount(1)
            .setPrimitiveOffset(0)
            .setFirstVertex(0)
            .setTransformOffset(0);

        // ビルドコマンドを送信してデバイス上でASをビルドする
        auto commandBuffer = vkutils::createCommandBuffer(device.get(), commandPool.get(), true);
        commandBuffer->buildAccelerationStructuresKHR(accelerationBuildGeometryInfo,
                                                      &accelerationStructureBuildRangeInfo);
        vkutils::submitCommandBuffer(device.get(), commandBuffer.get(), graphicsQueue);

        // Bottom Level AS のハンドルを取得する
        tlas.buffer.deviceAddress = device->getAccelerationStructureAddressKHR({tlas.handle.get()});
    }

    void createRayTracingPipeLine() {
        // Top Level ASをRaygenシェーダにバインドするための設定 [0]
        vk::DescriptorSetLayoutBinding accelerationStructureLayoutBinding{};
        accelerationStructureLayoutBinding.setBinding(0)
            .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

        // StorageImageをRaygenシェーダにバインドするための設定 [1]
        vk::DescriptorSetLayoutBinding resultImageLayoutBinding{};
        resultImageLayoutBinding.setBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

        // ディスクリプタセットレイアウトを作成する
        std::vector<vk::DescriptorSetLayoutBinding> binding{accelerationStructureLayoutBinding,
                                                            resultImageLayoutBinding};
        vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
        descriptorSetLayoutCreateInfo.setFlags(
            vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
        descriptorSetLayoutCreateInfo.setBindings(binding);
        descriptorSetLayout =
            device->createDescriptorSetLayoutUnique(descriptorSetLayoutCreateInfo);

        // パイプラインレイアウトを作成する
        pipelineLayout = device->createPipelineLayoutUnique(
            vk::PipelineLayoutCreateInfo{}.setSetLayouts(descriptorSetLayout.get()));

        // レイトレーシングシェーダグループの設定
        // 各シェーダグループはパイプライン内の対応するシェーダを指す
        std::array<vk::PipelineShaderStageCreateInfo, 3> shaderStages;
        constexpr uint32_t shaderIndexRaygen = 0;
        constexpr uint32_t shaderIndexMiss = 1;
        constexpr uint32_t shaderIndexClosestHit = 2;

        std::vector<vk::UniqueShaderModule> shaderModules;

        // Ray generation グループ
        shaderModules.push_back(
            vkutils::createShaderModule(device.get(), SHADER_DIR + "raygen.rgen.spv"));
        shaderStages[shaderIndexRaygen] = vk::PipelineShaderStageCreateInfo{}
                                              .setStage(vk::ShaderStageFlagBits::eRaygenKHR)
                                              .setModule(shaderModules.back().get())
                                              .setPName("main");
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{}
                                   .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
                                   .setGeneralShader(shaderIndexRaygen)
                                   .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                                   .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                                   .setIntersectionShader(VK_SHADER_UNUSED_KHR));

        // Ray miss グループ
        shaderModules.push_back(
            vkutils::createShaderModule(device.get(), SHADER_DIR + "miss.rmiss.spv"));
        shaderStages[shaderIndexMiss] = vk::PipelineShaderStageCreateInfo{}
                                            .setStage(vk::ShaderStageFlagBits::eMissKHR)
                                            .setModule(shaderModules.back().get())
                                            .setPName("main");
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{}
                                   .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
                                   .setGeneralShader(shaderIndexMiss)
                                   .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                                   .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                                   .setIntersectionShader(VK_SHADER_UNUSED_KHR));

        // Ray closest hit グループ
        shaderModules.push_back(
            vkutils::createShaderModule(device.get(), SHADER_DIR + "closesthit.rchit.spv"));
        shaderStages[shaderIndexClosestHit] = vk::PipelineShaderStageCreateInfo{}
                                                  .setStage(vk::ShaderStageFlagBits::eClosestHitKHR)
                                                  .setModule(shaderModules.back().get())
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
                                                          .setLayout(pipelineLayout.get()));
        if (result.result == vk::Result::eSuccess) {
            pipeline = std::move(result.value);
        } else {
            throw std::runtime_error("failed to create ray tracing pipeline.");
        }
    }

    void createShaderBindingTable() {
        const uint32_t handleSize = vkutils::getShaderGroupHandleSize();
        const uint32_t handleSizeAligned = vkutils::getHandleSizeAligned();
        const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
        const uint32_t sbtSize = groupCount * handleSizeAligned;

        const vk::BufferUsageFlags sbtBufferUsageFlags =
            vk::BufferUsageFlagBits::eShaderBindingTableKHR |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress;

        const vk::MemoryPropertyFlags sbtMemoryProperty =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        // シェーダグループのハンドルを取得する
        std::vector<uint8_t> shaderHandleStorage(sbtSize);
        auto result = device->getRayTracingShaderGroupHandlesKHR(
            pipeline.get(), 0, groupCount, sbtSize, shaderHandleStorage.data());
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to get ray tracing shader group handles.");
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
        descriptorPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                                          vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
        descriptorPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);

        // ディスクリプタセットを1つ準備する
        auto descriptorSets =
            device->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo{}
                                                     .setDescriptorPool(descriptorPool.get())
                                                     .setSetLayouts(descriptorSetLayout.get()));
        descriptorSet = std::move(descriptorSets.front());
    }

    void updateDescriptorSets(vk::ImageView imageView) {
        // Top Level ASをシェーダにバインドするためのディスクリプタ
        vk::WriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
        descriptorAccelerationStructureInfo.setAccelerationStructures(tlas.handle.get());

        vk::WriteDescriptorSet accelerationStructureWrite{};
        accelerationStructureWrite.setDstSet(descriptorSet.get())
            .setDstBinding(0)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
            .setPNext(&descriptorAccelerationStructureInfo);

        // Storage imageのためのディスクリプタ
        vk::DescriptorImageInfo imageDescriptor{};
        imageDescriptor.setImageView(imageView).setImageLayout(vk::ImageLayout::eGeneral);

        vk::WriteDescriptorSet resultImageWrite{};
        resultImageWrite.setDstSet(descriptorSet.get())
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

        const uint32_t handleSizeAligned = vkutils::getHandleSizeAligned();

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

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline.get());

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR,  // pipelineBindPoint
            pipelineLayout.get(),                   // layout
            0,                                      // firstSet
            descriptorSet.get(),                    // descriptorSets
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
        auto result = device->acquireNextImageKHR(swapchain.get(),  // swapchain
                                                  std::numeric_limits<uint64_t>::max(),  // timeout
                                                  imageAvailableSemaphore.get());
        uint32_t imageIndex;
        if (result.result == vk::Result::eSuccess) {
            imageIndex = result.value;
        } else {
            throw std::runtime_error("failed to acquire next image!");
        }

        updateDescriptorSets(swapchainImageViews[imageIndex].get());

        buildCommandBuffers(drawCommandBuffers[imageIndex].get(), swapchainImages[imageIndex]);

        // レイトレーシングを行うコマンドバッファを実行する
        vk::PipelineStageFlags waitStage{vk::PipelineStageFlagBits::eRayTracingShaderKHR};
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(drawCommandBuffers[imageIndex].get());
        submitInfo.setWaitSemaphores(imageAvailableSemaphore.get());
        graphicsQueue.submit(submitInfo);

        graphicsQueue.waitIdle();

        // 表示する
        graphicsQueue.presentKHR(
            vk::PresentInfoKHR{}.setSwapchains(swapchain.get()).setImageIndices(imageIndex));

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
        auto memoryRequirements = device->getBufferMemoryRequirements(buffer.handle.get());
        vk::MemoryAllocateFlagsInfo memoryFlagsInfo{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            memoryFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        }

        buffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
                .setAllocationSize(memoryRequirements.size)
                .setMemoryTypeIndex(vkutils::getMemoryType(memoryRequirements, memoryProperty))
                .setPNext(&memoryFlagsInfo));
        device->bindBufferMemory(buffer.handle.get(), buffer.deviceMemory.get(), 0);

        // データをメモリにコピーする
        if (data) {
            void* dataPtr = device->mapMemory(buffer.deviceMemory.get(), 0, size);
            memcpy(dataPtr, data, size);
            device->unmapMemory(buffer.deviceMemory.get());
        }

        // バッファのデバイスアドレスを取得する
        buffer.deviceAddress = getBufferDeviceAddress(buffer.handle.get());

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
        auto memoryRequirements = device->getBufferMemoryRequirements(buffer.handle.get());
        vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{
            vk::MemoryAllocateFlagBits::eDeviceAddress};

        buffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
                .setAllocationSize(memoryRequirements.size)
                .setMemoryTypeIndex(vkutils::getMemoryType(
                    memoryRequirements, vk::MemoryPropertyFlagBits::eHostVisible |
                                            vk::MemoryPropertyFlagBits::eHostCoherent))
                .setPNext(&memoryAllocateFlagsInfo));
        device->bindBufferMemory(buffer.handle.get(), buffer.deviceMemory.get(), 0);

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
        auto memoryRequirements = device->getBufferMemoryRequirements(scratchBuffer.handle.get());
        vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{
            vk::MemoryAllocateFlagBits::eDeviceAddress};

        scratchBuffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
                .setAllocationSize(memoryRequirements.size)
                .setMemoryTypeIndex(vkutils::getMemoryType(
                    memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal))
                .setPNext(&memoryAllocateFlagsInfo));
        device->bindBufferMemory(scratchBuffer.handle.get(), scratchBuffer.deviceMemory.get(), 0);

        // バッファのデバイスアドレスを取得する
        scratchBuffer.deviceAddress = getBufferDeviceAddress(scratchBuffer.handle.get());

        return scratchBuffer;
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        device->waitIdle();
    }

    void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};
