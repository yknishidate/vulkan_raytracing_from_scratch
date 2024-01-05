@echo off
set GLSLANG_VALIDATOR=%VULKAN_SDK%/Bin/glslangValidator.exe

for %%s in (raygen.rgen closesthit.rchit miss.rmiss) do (
    %GLSLANG_VALIDATOR% %%s -V -o %%s.spv --target-env vulkan1.2
)
