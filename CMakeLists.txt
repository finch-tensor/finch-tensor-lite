cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" FINCHLITE_PYPROJECT)
string(REGEX MATCH "version = \"([^\"]+)\"" _ "${FINCHLITE_PYPROJECT}")
set(FINCHLITE_VERSION "${CMAKE_MATCH_1}")
if(NOT FINCHLITE_VERSION)
  message(FATAL_ERROR "Could not read project version from pyproject.toml")
endif()

project(finch_tensor_lite VERSION "${FINCHLITE_VERSION}" LANGUAGES NONE)

find_package(Python REQUIRED COMPONENTS Interpreter)

if(NOT DEFINED ENV{PREFIX})
  message(FATAL_ERROR "PREFIX must be set by the Pixi package build environment")
endif()

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import os, sysconfig; print(os.path.relpath(sysconfig.get_paths()['purelib'], os.environ['PREFIX']))"
  OUTPUT_VARIABLE FINCHLITE_PURELIB
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/finchlite"
  DESTINATION "${FINCHLITE_PURELIB}"
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "py.typed"
  PATTERN "__pycache__" EXCLUDE
)

set(FINCHLITE_DIST_INFO "finch_tensor_lite-${PROJECT_VERSION}.dist-info")
set(FINCHLITE_DIST_INFO_DIR "${CMAKE_CURRENT_BINARY_DIR}/${FINCHLITE_DIST_INFO}")

file(MAKE_DIRECTORY "${FINCHLITE_DIST_INFO_DIR}")
file(
  WRITE "${FINCHLITE_DIST_INFO_DIR}/METADATA"
  "Metadata-Version: 2.4\n"
  "Name: finch-tensor-lite\n"
  "Version: ${PROJECT_VERSION}\n"
)
file(
  WRITE "${FINCHLITE_DIST_INFO_DIR}/WHEEL"
  "Wheel-Version: 1.0\n"
  "Generator: pixi-build-cmake\n"
  "Root-Is-Purelib: true\n"
  "Tag: py3-none-any\n"
)
file(WRITE "${FINCHLITE_DIST_INFO_DIR}/RECORD" "")

install(
  DIRECTORY "${FINCHLITE_DIST_INFO_DIR}/"
  DESTINATION "${FINCHLITE_PURELIB}/${FINCHLITE_DIST_INFO}"
)
