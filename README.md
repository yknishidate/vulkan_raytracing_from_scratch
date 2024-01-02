# Vulkan Ray Tracing from Scratch

Vulkan Ray Tracing を使って三角形をレンダリングするプロジェクト

![triangle](triangle.png)

## Features

- サードパーティの Vulkan ライブラリを使用せずゼロから記述する
- 公式 C++ラッパー `vulkan.hpp` を使うため冗長な記述が少ない
- 小さなヘルパーヘッダを 1 つだけ提供する
- 各ステップごとにプログラムを残しておく

## Requirement

- Vulkan Ray TracingをサポートするGPUとドライバ
- Vulkan SDK 1.2.162.0 or later
- C++17
- CMake
- vcpkg

## Generate project

```sh
# Make sure that VCPKG_ROOT is set
cmake . -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
```
