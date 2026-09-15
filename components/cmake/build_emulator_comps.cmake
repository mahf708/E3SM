# Emulator Components build configuration
# Handles all emulator components: EMULATORATM, EMULATOROCN, EMULATORICE
# Similar pattern to build_eamxx.cmake

# Capture this file's directory at include-time.
# CMAKE_CURRENT_LIST_DIR inside a function body is evaluated at *call* time
# (i.e. it becomes the caller's directory), so we must snapshot it here.
set(_EMULATOR_BUILD_COMPS_DIR ${CMAKE_CURRENT_LIST_DIR})

function(build_emulator_comps)

  # The emulated components this case uses: emulatoratm, emulatorocn,
  # emulatorice.
  set(_EMULATOR_COMP_NAMES "")
  foreach(_comp IN LISTS COMP_NAMES)
    if(_comp MATCHES "^emulator")
      list(APPEND _EMULATOR_COMP_NAMES ${_comp})
    endif()
  endforeach()

  if (_EMULATOR_COMP_NAMES)

    message(STATUS "")
    message(STATUS "=================================================================")
    message(STATUS "  Building Emulator Components Framework")
    message(STATUS "=================================================================")
    message(STATUS "  Components: ${_EMULATOR_COMP_NAMES}")

    include(${_EMULATOR_BUILD_COMPS_DIR}/common_setup.cmake)

    # The ML backend a case runs is chosen in its input files; whether
    # LibTorch is built in is decided here, by the machine's environment.
    if(DEFINED ENV{Torch_ROOT} AND NOT DEFINED EMULATOR_ENABLE_LIBTORCH)
      set(EMULATOR_ENABLE_LIBTORCH ON)
    endif()

    #---------------------------------------------------------------------------
    # Build emulator_comps using add_subdirectory
    #---------------------------------------------------------------------------
    set(EMULATOR_COMPS_DIR ${_EMULATOR_BUILD_COMPS_DIR}/../emulators)
    if(EXISTS ${EMULATOR_COMPS_DIR}/CMakeLists.txt)
      message(STATUS "  Including emulator components from:")
      message(STATUS "    ${EMULATOR_COMPS_DIR}")
      add_subdirectory(${EMULATOR_COMPS_DIR} ${CMAKE_BINARY_DIR}/emulators)
    else()
      message(FATAL_ERROR
        "Emulator components directory not found!\n"
        "  Expected: ${EMULATOR_COMPS_DIR}/CMakeLists.txt\n"
        "  Current source dir: ${CMAKE_SOURCE_DIR}\n"
        "Please ensure the emulator components are present in the source tree.")
    endif()

    # build_model.cmake links each component class by its class name, so
    # each emulated component library stands in for that class's target.
    foreach(_comp IN LISTS _EMULATOR_COMP_NAMES)
      string(REGEX REPLACE "^emulator" "" _class ${_comp})
      add_library(${_class} ALIAS ${_comp})
    endforeach()

    # So CMakeLists.txt skips cmake/<class> for these components.
    set(EMULATOR_COMP_NAMES ${_EMULATOR_COMP_NAMES} PARENT_SCOPE)
    set(EMULATOR_COMP_NAMES ${_EMULATOR_COMP_NAMES} CACHE INTERNAL "List of emulator component names")

    message(STATUS "  Emulator components to skip in build_model: ${_EMULATOR_COMP_NAMES}")
    message(STATUS "=================================================================")
    message(STATUS "")
  endif()

endfunction(build_emulator_comps)
