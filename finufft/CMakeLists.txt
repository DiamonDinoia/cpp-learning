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
