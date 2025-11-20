# 最低 CMake 版本（与 ns-3.37 构建系统兼容的常见版本）
cmake_minimum_required(VERSION 3.16)

# 1) 找到 Python3 解释器 + 开发/嵌入组件（会提供 Python.h 与 libpython）
#    若你用的是 3.11/3.12，请在外部配置时用 -DPython3_EXECUTABLE=/usr/bin/python3.11 指定
find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module Development.Embed)

# 2) 源码/头文件清单（保持你的原始列表不变）
set(qoar_source_files
  helper/qoar-helper.cc
  model/qoar-dpd.cc
  model/qoar-id-cache.cc
  model/qoar-neighbor.cc
  model/qoar-packet.cc
  model/qoar-routing-protocol.cc
  model/qoar-rqueue.cc
  model/qoar-rtable.cc
  model/citr-adhoc-wifi-mac.cc
)

set(qoar_header_files
  helper/qoar-helper.h
  model/qoar-dpd.h
  model/qoar-id-cache.h
  model/qoar-neighbor.h
  model/qoar-packet.h
  model/qoar-routing-protocol.h
  model/qoar-rqueue.h
  model/qoar-rtable.h
  model/qoar-qlearning.h
  model/citr-adhoc-wifi-mac.h
)

# 3) 构建库：关键是把 Python3::Python (和可选的 Python3::Module) 放进 LIBRARIES_TO_LINK
#    ${libcore}/${libnetwork}/${libinternet}/${libwifi} 由 ns-3 顶层提供
build_lib(
  LIBNAME qoar
  SOURCE_FILES
    ${qoar_source_files}
  HEADER_FILES
    ${qoar_header_files}
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libwifi}
    Python3::Python
    Python3::Module
  TEST_SOURCES
    test/qoar-id-cache-test-suite.cc
    test/qoar-regression.cc
    test/qoar-test-suite.cc
    test/loopback.cc
    test/bug-772.cc
)

# 可选：如果你的模块里需要导出额外的 include，可在此追加（通常不需要）
# target_include_directories(ns3-qoar PRIVATE ${Python3_INCLUDE_DIRS})

