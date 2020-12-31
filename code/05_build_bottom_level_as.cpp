
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

struct Vertex
{
    float pos[3];
};

struct Buffer
{
    vk::UniqueBuffer handle;
    vk::UniqueDeviceMemory deviceMemory;
    uint64_t deviceAddress;
};

struct AccelerationStructure
{
    vk::UniqueAccelerationStructureKHR handle;
    Buffer buffer;
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

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> drawCommandBuffers;

    StorageImage storageImage;

    AccelerationStructure blas;

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

        createStorageImage();
        createBottomLevelAS();
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

    void createBottomLevelAS()
    {
        // 三角形のデータを用意
        std::vector<Vertex> vertices = {
            {{1.0f, 1.0f, 0.0f}},
            {{-1.0f, 1.0f, 0.0f}},
            {{0.0f, -1.0f, 0.0f}} };
        std::vector<uint32_t> indices = { 0, 1, 2 };

        // データからバッファを作成
        auto vertexBufferSize = vertices.size() * sizeof(Vertex);
        auto indexBufferSize = indices.size() * sizeof(uint32_t);
        vk::BufferUsageFlags bufferUsage{ vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
                                        | vk::BufferUsageFlagBits::eShaderDeviceAddress
                                        | vk::BufferUsageFlagBits::eStorageBuffer };
        vk::MemoryPropertyFlags memoryProperty{ vk::MemoryPropertyFlagBits::eHostVisible
                                              | vk::MemoryPropertyFlagBits::eHostCoherent };
        Buffer vertexBuffer = createBuffer(vertexBufferSize, bufferUsage, memoryProperty, vertices.data());
        Buffer indexBuffer = createBuffer(indexBufferSize, bufferUsage, memoryProperty, indices.data());

        // ジオメトリには三角形データを渡す
        vk::AccelerationStructureGeometryTrianglesDataKHR triangleData{};
        triangleData
            .setVertexFormat(vk::Format::eR32G32B32Sfloat)
            .setVertexData(vertexBuffer.deviceAddress)
            .setVertexStride(sizeof(Vertex))
            .setMaxVertex(vertices.size())
            .setIndexType(vk::IndexType::eUint32)
            .setIndexData(indexBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry
            .setGeometryType(vk::GeometryTypeKHR::eTriangles)
            .setGeometry({ triangleData })
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // ASビルドに必要なサイズを取得する
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setGeometries(geometry);

        const uint32_t primitiveCount = 1;
        auto buildSizesInfo = device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);

        // ASを保持するためのバッファを作成する
        blas.buffer = createAccelerationStructureBuffer(buildSizesInfo);

        // ASを作成する
        blas.handle = device->createAccelerationStructureKHRUnique(
            vk::AccelerationStructureCreateInfoKHR{}
            .setBuffer(blas.buffer.handle.get())
            .setSize(buildSizesInfo.accelerationStructureSize)
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
        );

        // ここから実際のビルドを行っていく

        // スクラッチバッファを作成する
        Buffer scratchBuffer = createScratchBuffer(buildSizesInfo.buildScratchSize);

        // ビルド情報を作成する
        vk::AccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{};
        accelerationBuildGeometryInfo
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
            .setDstAccelerationStructure(blas.handle.get())
            .setGeometries(geometry)
            .setScratchData(scratchBuffer.deviceAddress);

        vk::AccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
        accelerationStructureBuildRangeInfo
            .setPrimitiveCount(1)
            .setPrimitiveOffset(0)
            .setFirstVertex(0)
            .setTransformOffset(0);

        // ビルドコマンドを送信してデバイス上でASをビルドする
        auto commandBuffer = vkutils::createCommandBuffer(device.get(), commandPool.get(), true);
        commandBuffer->buildAccelerationStructuresKHR(accelerationBuildGeometryInfo, &accelerationStructureBuildRangeInfo);
        vkutils::submitCommandBuffer(device.get(), commandBuffer.get(), graphicsQueue);

        // Bottom Level AS のハンドルを取得する
        blas.buffer.deviceAddress = device->getAccelerationStructureAddressKHR({ blas.handle.get() });
    }

    Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryPropertiy, void* data = nullptr)
    {
        // Bufferオブジェクトを作成
        Buffer buffer{};
        buffer.handle = device->createBufferUnique(
            vk::BufferCreateInfo{}
            .setSize(size)
            .setUsage(usage)
            .setQueueFamilyIndexCount(0)
        );

        // メモリを確保してバインドする
        auto memoryRequirements = device->getBufferMemoryRequirements(buffer.handle.get());
        vk::MemoryAllocateFlagsInfo memoryFlagsInfo{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            memoryFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        }

        buffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
            .setAllocationSize(memoryRequirements.size)
            .setMemoryTypeIndex(vkutils::getMemoryType(memoryRequirements, memoryPropertiy))
            .setPNext(&memoryFlagsInfo)
        );
        device->bindBufferMemory(buffer.handle.get(), buffer.deviceMemory.get(), 0);

        // データをメモリにコピーする
        if (data) {
            void* dataPtr = device->mapMemory(buffer.deviceMemory.get(), 0, size);
            memcpy(dataPtr, data, static_cast<size_t>(size));
            device->unmapMemory(buffer.deviceMemory.get());
        }

        // バッファのデバイスアドレスを取得する
        vk::BufferDeviceAddressInfoKHR bufferDeviceAddressInfo{ buffer.handle.get() };
        buffer.deviceAddress = getBufferDeviceAddress(buffer.handle.get());

        return buffer;
    }

    uint64_t getBufferDeviceAddress(vk::Buffer buffer)
    {
        vk::BufferDeviceAddressInfoKHR bufferDeviceAI{ buffer };
        return device->getBufferAddressKHR(&bufferDeviceAI);
    }

    Buffer createAccelerationStructureBuffer(vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo)
    {
        // Bufferオブジェクトを作成
        Buffer buffer{};
        buffer.handle = device->createBufferUnique(
            vk::BufferCreateInfo{}
            .setSize(buildSizesInfo.accelerationStructureSize)
            .setUsage(vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR
                | vk::BufferUsageFlagBits::eShaderDeviceAddress)
        );

        // メモリを確保してバインドする
        auto memoryRequirements = device->getBufferMemoryRequirements(buffer.handle.get());
        vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };

        buffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
            .setAllocationSize(memoryRequirements.size)
            .setMemoryTypeIndex(vkutils::getMemoryType(
                memoryRequirements, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))
            .setPNext(&memoryAllocateFlagsInfo)
        );
        device->bindBufferMemory(buffer.handle.get(), buffer.deviceMemory.get(), 0);

        return buffer;
    }

    Buffer createScratchBuffer(vk::DeviceSize size)
    {
        Buffer scratchBuffer;

        // バッファを作成する
        scratchBuffer.handle = device->createBufferUnique(
            vk::BufferCreateInfo{}
            .setSize(size)
            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress)
        );

        // メモリを確保してバインドする
        auto memoryRequirements = device->getBufferMemoryRequirements(scratchBuffer.handle.get());
        vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };

        scratchBuffer.deviceMemory = device->allocateMemoryUnique(
            vk::MemoryAllocateInfo{}
            .setAllocationSize(memoryRequirements.size)
            .setMemoryTypeIndex(vkutils::getMemoryType(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal))
            .setPNext(&memoryAllocateFlagsInfo)
        );
        device->bindBufferMemory(scratchBuffer.handle.get(), scratchBuffer.deviceMemory.get(), 0);

        // バッファのデバイスアドレスを取得する
        vk::BufferDeviceAddressInfoKHR bufferDeviceAddressInfo{ scratchBuffer.handle.get() };
        scratchBuffer.deviceAddress = getBufferDeviceAddress(scratchBuffer.handle.get());

        return scratchBuffer;
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
