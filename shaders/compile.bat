@echo off
set GLSLANG_VALIDATOR=%VULKAN_SDK%/Bin/glslangValidator.exe

for %%s in (*.rgen *.rchit *.rmiss) do (
    %GLSLANG_VALIDATOR% %%s -V -o %%s.spv --target-env vulkan1.2
)
