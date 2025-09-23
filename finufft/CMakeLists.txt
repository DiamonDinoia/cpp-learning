include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found. Enable CPP_LEARNING_ENABLE_FINUFFT only when a CUDA toolchain is available.")
endif()

enable_language(CUDA)

include(FetchContent)

set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)
set(FINUFFT_USE_CPU OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        FINUFFT
        GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
        GIT_TAG master # or master
)

FetchContent_MakeAvailable(FINUFFT)

add_executable(example_cufinufft example2d1many.cu)
target_link_libraries(example_cufinufft cufinufft)
set_target_properties(example_cufinufft PROPERTIES CUDA_ARCHITECTURES "native")
