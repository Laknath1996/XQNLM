# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/build"

# Include any dependencies generated for this target.
include CMakeFiles/add_noise.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/add_noise.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/add_noise.dir/flags.make

CMakeFiles/add_noise.dir/add_noise.cxx.o: CMakeFiles/add_noise.dir/flags.make
CMakeFiles/add_noise.dir/add_noise.cxx.o: ../add_noise.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/add_noise.dir/add_noise.cxx.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/add_noise.dir/add_noise.cxx.o -c "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/add_noise.cxx"

CMakeFiles/add_noise.dir/add_noise.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/add_noise.dir/add_noise.cxx.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/add_noise.cxx" > CMakeFiles/add_noise.dir/add_noise.cxx.i

CMakeFiles/add_noise.dir/add_noise.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/add_noise.dir/add_noise.cxx.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/add_noise.cxx" -o CMakeFiles/add_noise.dir/add_noise.cxx.s

# Object files for target add_noise
add_noise_OBJECTS = \
"CMakeFiles/add_noise.dir/add_noise.cxx.o"

# External object files for target add_noise
add_noise_EXTERNAL_OBJECTS =

add_noise: CMakeFiles/add_noise.dir/add_noise.cxx.o
add_noise: CMakeFiles/add_noise.dir/build.make
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkdouble-conversion-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitksys-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkvnl_algo-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkvnl-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkv3p_netlib-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitknetlib-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkvcl-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKCommon-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkNetlibSlatec-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKStatistics-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKTransform-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKLabelMap-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKMesh-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkzlib-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKMetaIO-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKSpatialObjects-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKPath-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKQuadEdgeMesh-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOImageBase-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKOptimizers-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKPolynomials-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKBiasCorrection-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKBioCell-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKDICOMParser-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKEXPAT-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOXML-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOSpatialObjects-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKFEM-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmDICT-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmMSFF-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKznz-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKniftiio-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKgiftiio-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkhdf5_cpp.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkhdf5.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOBMP-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOBioRad-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOBruker-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOCSV-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOGDCM-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOIPL-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOGE-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOGIPL-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOHDF5-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkjpeg-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOJPEG-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitktiff-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOTIFF-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOLSM-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkminc2-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOMINC-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOMRC-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOMesh-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOMeta-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIONIFTI-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKNrrdIO-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIONRRD-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkpng-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOPNG-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOSiemens-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOStimulate-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKTransformFactory-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOTransformBase-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOTransformHDF5-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOTransformInsightLegacy-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOTransformMatlab-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOVTK-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKKLMRegionGrowing-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitklbfgs-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKOptimizersv4-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKVTK-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKVideoCore-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKVideoIO-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKWatersheds-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOXML-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmMSFF-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmDICT-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmIOD-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmDSED-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmCommon-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmjpeg8-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmjpeg12-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmjpeg16-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmopenjp2-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmcharls-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkgdcmuuid-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOTIFF-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitktiff-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkjpeg-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkminc2-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKgiftiio-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKEXPAT-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKMetaIO-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKniftiio-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKznz-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKNrrdIO-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkpng-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOIPL-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkhdf5_cpp.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkhdf5.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkzlib-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOTransformBase-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKTransformFactory-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKOptimizers-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitklbfgs-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKIOImageBase-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKVideoCore-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKStatistics-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkNetlibSlatec-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKSpatialObjects-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKMesh-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKTransform-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKPath-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKCommon-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkdouble-conversion-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitksys-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libITKVNLInstantiation-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkvnl_algo-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkvnl-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkv3p_netlib-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitknetlib-4.13.a
add_noise: /Users/ashwin/Semester\ 7/ITK/ITKbin/lib/libitkvcl-4.13.a
add_noise: CMakeFiles/add_noise.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable add_noise"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add_noise.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/add_noise.dir/build: add_noise

.PHONY : CMakeFiles/add_noise.dir/build

CMakeFiles/add_noise.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/add_noise.dir/cmake_clean.cmake
.PHONY : CMakeFiles/add_noise.dir/clean

CMakeFiles/add_noise.dir/depend:
	cd "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation" "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation" "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/build" "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/build" "/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/data/example/itk_implementation/build/CMakeFiles/add_noise.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/add_noise.dir/depend

