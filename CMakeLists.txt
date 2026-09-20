cmake_minimum_required(VERSION 3.28)

# Phase 1 supports the explicit mixed image 86;120a plus the inherited
# homogeneous forms. Validate before CUDA compiler detection so unsupported
# combinations fail closed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86;120a" CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89|86-real;86-virtual|86;120a)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=86, 89, 86-real;86-virtual, or 86;120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# NINFER_SM8X_COMPAT gates every Ampere/early-Ada compatibility path. ninfer_ops also sets it
# PUBLIC for its own consumers, but that link is PRIVATE through ninfer_engine, so apps and tests
# would not otherwise see it. Defining it globally keeps a single consistent answer for every
# target, including apps/ninfer-perplexity, whose upstream default KV profile is FP8.
set(NINFER_SM8X_COMPAT_VALUE 0)
if(CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)(86|89)(-real|-virtual)?(;|$)")
  add_compile_definitions(NINFER_SM8X_COMPAT=1)
  set(NINFER_SM8X_COMPAT_VALUE 1)
endif()

# Build-type default MUST be resolved before the M5.4b fingerprint block below: the fingerprint
# records CMAKE_BUILD_TYPE, so a fresh single-config build without an explicit
# -DCMAKE_BUILD_TYPE would otherwise generate a header with an empty build_type and then compile
# as Release anyway -- silently mismatching the identity it just fingerprinted.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# M5.4b build identity: byte-stable fingerprint of kernel-affecting build inputs (see
# src/targets/calibration_build_id.h.in for the exact input list and rationale). Sorted so list
# order alone never perturbs the fingerprint.
set(NINFER_SORTED_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
list(SORT NINFER_SORTED_CUDA_ARCHITECTURES)
list(JOIN NINFER_SORTED_CUDA_ARCHITECTURES "," NINFER_SORTED_CUDA_ARCHITECTURES)

# CMAKE_CXX_FLAGS_<CONFIG>/CMAKE_CUDA_FLAGS_<CONFIG> are per-config variables that
# configure_file(@ONLY) cannot expand (it does not evaluate nested ${...} references inside a
# substituted value) -- resolve the active config's variable names into flat variables first.
string(TOUPPER "${CMAKE_BUILD_TYPE}" NINFER_BUILD_TYPE_UPPER)
set(NINFER_ACTIVE_CXX_FLAGS_CONFIG "${CMAKE_CXX_FLAGS_${NINFER_BUILD_TYPE_UPPER}}")
set(NINFER_ACTIVE_CUDA_FLAGS_CONFIG "${CMAKE_CUDA_FLAGS_${NINFER_BUILD_TYPE_UPPER}}")

# Escape backslashes THEN double-quotes (order matters: escaping quotes first would re-escape the
# backslashes just inserted) so every flag value substitutes into calibration_build_id.h.in as a
# valid C++ string literal even if it itself contains a `"` or `\`.
function(ninfer_escape_for_cxx_string_literal input_var)
  set(_value "${${input_var}}")
  string(REPLACE "\\" "\\\\" _value "${_value}")
  string(REPLACE "\"" "\\\"" _value "${_value}")
  set(${input_var} "${_value}" PARENT_SCOPE)
endfunction()

ninfer_escape_for_cxx_string_literal(CMAKE_CXX_FLAGS)
ninfer_escape_for_cxx_string_literal(NINFER_ACTIVE_CXX_FLAGS_CONFIG)
ninfer_escape_for_cxx_string_literal(CMAKE_CUDA_FLAGS)
ninfer_escape_for_cxx_string_literal(NINFER_ACTIVE_CUDA_FLAGS_CONFIG)

# Deterministic audit representation of the effective extra compile flags (empty only when
# genuinely no extra flags are set for this build) -- populates BuildIdentity::compile_flags.
set(NINFER_COMPILE_FLAGS_AUDIT "")
foreach(_flag_pair
    "cxx_flags=${CMAKE_CXX_FLAGS}"
    "cxx_flags_config=${NINFER_ACTIVE_CXX_FLAGS_CONFIG}"
    "cuda_flags=${CMAKE_CUDA_FLAGS}"
    "cuda_flags_config=${NINFER_ACTIVE_CUDA_FLAGS_CONFIG}")
  if(NOT _flag_pair MATCHES "=$")
    list(APPEND NINFER_COMPILE_FLAGS_AUDIT "${_flag_pair}")
  endif()
endforeach()
list(JOIN NINFER_COMPILE_FLAGS_AUDIT ";" NINFER_COMPILE_FLAGS_AUDIT)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/targets/calibration_build_id.h.in
  ${PROJECT_BINARY_DIR}/generated/targets/calibration_build_id.h
  @ONLY)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # /Zc:__cplusplus makes MSVC report the standard it is actually compiling. Without it the macro
  # stays at 199711L even under /std:c++20, and CUDA's <vector_types.h> then hides the constexpr
  # dim3 constructors behind its own __cplusplus >= 201103L test, which fails to compile every
  # translation unit that builds a dim3 in a constant expression.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process.
  add_link_options(/NODEFAULTLIB:LIBCMT)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
