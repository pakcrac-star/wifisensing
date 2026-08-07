cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(wifisensing)

# Define Rust engine paths and target architecture
set(RUST_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/main/rust_engine")
set(RUST_TARGET "xtensa-esp32s3-none-elf")
set(RUST_TARGET_DIR "D:/rt")
set(RUST_LIB_PATH "${RUST_TARGET_DIR}/${RUST_TARGET}/release/librust_engine.a")

# Custom target to compile the `no_std` Rust static library via Cargo before linking
add_custom_target(rust_engine ALL
    COMMAND cargo build --release --target ${RUST_TARGET} --target-dir ${RUST_TARGET_DIR}
    WORKING_DIRECTORY ${RUST_ENGINE_DIR}
    COMMENT "Compiling Rust DSP Engine..."
)
