cmake_minimum_required(VERSION 3.16)
project(SeisCL C)

set(CMAKE_C_STANDARD 11)


option(USEMPI "Compile with MPI support" ON)
option(USECUDA "Use CUDA instead of OpenCL" OFF)
set(CMAKE_BUILD_TYPE Debug)

if (USEMPI)
    add_executable(SeisCL_MPI
            src/third_party/KISS_FFT/kiss_fft.c
            src/third_party/KISS_FFT/kiss_fftr.c
            src/third_party/SOFI3D/CPML_COEFF.c
            src/third_party/SOFI3D/holbergcoeff.c
            src/assign_modeling_case.c
            src/automatic_kernels.c
            src/butterworth.c
            src/calc_grad.c
            src/clbuf.c
            src/clerrors.c
            src/clmodel.c
            src/clprogram.c
            src/comm.c
            src/event_dependency.c
            src/Free_OpenCL.c
            src/Init_cst.c
            src/Init_data.c
            src/Init_model.c
            src/Init_MPI.c
            src/Init_OpenCL.c
            src/Out_MPI.c
            src/read_hdf5.c
            src/residuals.c
            src/SeisCL_MPI.c
            src/time_stepping.c
            src/writehdf5.c)
else (USEMPI)
    add_executable(SeisCL_MPI
            src/third_party/KISS_FFT/kiss_fft.c
            src/third_party/KISS_FFT/kiss_fftr.c
            src/third_party/SOFI3D/CPML_COEFF.c
            src/third_party/SOFI3D/holbergcoeff.c
            src/assign_modeling_case.c
            src/automatic_kernels.c
            src/butterworth.c
            src/calc_grad.c
            src/clbuf.c
            src/clerrors.c
            src/clmodel.c
            src/clprogram.c
            src/comm.c
            src/event_dependency.c
            src/Free_OpenCL.c
            src/Init_cst.c
            src/Init_data.c
            src/Init_model.c
            src/Init_OpenCL.c
            src/read_hdf5.c
            src/residuals.c
            src/SeisCL_MPI.c
            src/time_stepping.c
            src/writehdf5.c)
endif(USEMPI)

find_package(HDF5 COMPONENTS C HL REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
target_link_libraries(SeisCL_MPI ${HDF5_LIBRARIES})
target_link_libraries(SeisCL_MPI ${HDF5_HL_LIBRARIES})


if (USECUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(__CUDA_INCLUDE__="${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(SeisCL_MPI CUDA::cuda_driver)
    target_link_libraries(SeisCL_MPI CUDA::nvrtc)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
else(USECUDA)
    find_package(OpenCL)
    if (OpenCL_FOUND)
        include_directories(${OpenCL_INCLUDE_DIRS})
        target_link_libraries(SeisCL_MPI ${OpenCL_LIBRARIES})
        add_compile_definitions(__SEISCL__)
    else(OpenCL_FOUND)
        message(STATUS "No OpenCL library support")
    endif (OpenCL_FOUND)
endif (USECUDA)


find_package(MPI)
if (MPI_FOUND AND USEMPI)
    include_directories(${MPI_C_INCLUDE_DIRS})
    target_link_libraries(SeisCL_MPI ${MPI_LIBRARIES})
    if(MPI_COMPILE_FLAGS)
        set_target_properties(SeisCL_MPI PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()
    if(MPI_LINK_FLAGS)
        set_target_properties(SeisCL_MPI PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
else(MPI_FOUND AND USEMPI)
    if (USEMPI)
        message(ERROR "Compiling without MPI support")
    else(USEMPI)
        message(STATUS "Compiling without MPI support")
        add_compile_definitions(__NOMPI__)
    endif(USEMPI)
endif (MPI_FOUND AND USEMPI)


include_directories(src)
include_directories(src/third_party/KISS_FFT)
include_directories(src/third_party/NVIDIA_FP16)
include_directories(src/third_party/SOFI3D)



add_executable(cl2cstring src/cl2cstring.c)
file(GLOB CLKERNELS src/*.cl)
foreach(filecl ${CLKERNELS})
    get_filename_component(filehcl1 ${filecl} DIRECTORY)
    get_filename_component(filehcl2 ${filecl} NAME_WE)
    add_custom_target(${filehcl2}convert
            DEPENDS
            "${filehcl1}/${filehcl2}.hcl"
            )
    add_custom_command(
            OUTPUT
            "${filehcl1}/${filehcl2}.hcl"
            COMMENT
            "This is generating my custom command"
            COMMAND
            cl2cstring ARGS ${filehcl2}.cl ${filehcl2}.hcl
            DEPENDS
            ${filecl}
            WORKING_DIRECTORY
            ${filehcl1}
    )
    add_dependencies(SeisCL_MPI ${filehcl2}convert)
endforeach(filecl)

MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Library Type: " ${LIB_TYPE})
MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})
