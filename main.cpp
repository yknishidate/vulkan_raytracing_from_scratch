
#include "vkutils.hpp"

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

    void initWindow()
    {

    }

    void initVulkan()
    {

    }


    void mainLoop()
    {

    }

    void cleanup()
    {

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
