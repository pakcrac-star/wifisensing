idf_component_register(SRCS "main.c" "wifi_csi.c" "telemetry_transport.c"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES esp_wifi 
                                  esp_netif 
                                  esp_event 
                                  esp_timer 
                                  nvs_flash 
                                  esp_ringbuf 
                                  driver 
                                  esp_driver_uart
                                  freertos 
                                  log 
                                  esp_system 
                                  esp_common)

# ------------------------------------------------------------------------------
# DIRECT RUST LINKING 
# ------------------------------------------------------------------------------

# Define the exact path to the .a file. 
# NOTE: Check your target architecture folder name! It is usually "xtensa-esp32s3-none-elf" for no_std.
set(RUST_LIB_FILE "${CMAKE_CURRENT_SOURCE_DIR}/rust_engine/target/xtensa-esp32s3-none-elf/release/librust_engine.a")

# Print the path during the build so you can verify it
message(STATUS "=======================================================")
message(STATUS ">>> LOOKING FOR RUST LIB AT: ${RUST_LIB_FILE}")
message(STATUS "=======================================================")

# Link the exact file directly instead of using search directories
target_link_libraries(${COMPONENT_LIB} PRIVATE ${RUST_LIB_FILE})
