/**
 * @file main.c
 * @brief System Orchestrator and Boot Menu for ESP32-S3 Wi-Fi Sensing.
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_netif.h"
#include "driver/uart.h"

#include "wifi_csi.h"
#include "telemetry_transport.h"

static const char *TAG __attribute__((unused)) = "ORCHESTRATOR";

/* =========================================================================
 * External FFI Declarations (Bridges to Rust & Telemetry)
 * ========================================================================= */

extern void rust_engine_init(void);
extern void rust_engine_push_csi(const csi_frame_t *frame);
extern void rust_engine_push_metadata(const metadata_frame_t *meta);

/* =========================================================================
 * Global State for Hybrid Mode Gating
 * ========================================================================= */

static volatile bool g_is_hybrid_mode = false;
static volatile bool g_hybrid_permission_granted = false;

/* =========================================================================
 * Serial Monitor I/O Helpers
 * ========================================================================= */

static void serial_readline(char *buf, size_t max_len)
{
    size_t count = 0;
    while (count < max_len - 1) {
        int c = getchar();
        if (c == '\n' || c == '\r') {
            if (count > 0) {
                break;
            }
        } else if (c != EOF) {
            buf[count++] = (char)c;
            putchar(c);
        } else {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }
    buf[count] = '\0';
    printf("\n");
}

/* =========================================================================
 * Wi-Fi Scanning & Selection Menu
 * ========================================================================= */

static void run_ap_scanner_menu(char *out_ssid, char *out_password)
{
    printf("\n--- Starting Temporary Wi-Fi Scanner ---\n");

    esp_netif_t *temp_netif = esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    wifi_scan_config_t scan_config = { .show_hidden = true };
    printf("Scanning nearby networks...\n");
    ESP_ERROR_CHECK(esp_wifi_scan_start(&scan_config, true));

    uint16_t ap_count = 0;
    esp_wifi_scan_get_ap_num(&ap_count);

    if (ap_count == 0) {
        printf("No APs found. Defaulting to empty SSID.\n");
        out_ssid[0] = '\0';
        out_password[0] = '\0';
    } else {
        wifi_ap_record_t *ap_records = malloc(sizeof(wifi_ap_record_t) * ap_count);
        if (ap_records) {
            ESP_ERROR_CHECK(esp_wifi_scan_get_ap_records(&ap_count, ap_records));

            printf("\nFound %d networks:\n", ap_count);
            for (int i = 0; i < ap_count; i++) {
                printf("[%d] %s (RSSI: %d, Ch: %d)\n", i, ap_records[i].ssid, ap_records[i].rssi, ap_records[i].primary);
            }

            char input_buf[32];
            int selection = -1;
            while (selection < 0 || selection >= ap_count) {
                printf("\nEnter the number of the Wi-Fi to connect to: ");
                serial_readline(input_buf, sizeof(input_buf));
                selection = atoi(input_buf);
            }

            strncpy(out_ssid, (char *)ap_records[selection].ssid, 32);
            out_ssid[32] = '\0';

            printf("Enter password for '%s': ", out_ssid);
            serial_readline(out_password, 64);

            free(ap_records);
        }
    }

    ESP_ERROR_CHECK(esp_wifi_stop());
    ESP_ERROR_CHECK(esp_wifi_deinit());
    if (temp_netif) {
        esp_netif_destroy_default_wifi(temp_netif);
    }
    
    printf("--- Scanner Teardown Complete ---\n\n");
}

/* =========================================================================
 * Data Router Task (Pinned safely to Core 1)
 * ========================================================================= */

static void data_router_task(void *pvParameters)
{
    static csi_frame_t csi;
    static metadata_frame_t meta;

    while (1) {
        bool active_work = false;

        // 1. Route High-Speed CSI
        if (wifi_csi_pop_frame(&csi, 10)) {
            rust_engine_push_csi(&csi);
            telemetry_transport_send(&csi, sizeof(csi));
            active_work = true;
        }

        // 2. Route Slow-Path Metadata
        if (wifi_csi_pop_metadata(&meta, 10)) {
            if (!g_is_hybrid_mode || g_hybrid_permission_granted) {
                rust_engine_push_metadata(&meta);
            }
            active_work = true;
        }

        // Yield control to prevent watchdog starvation and high CPU heat
        if (!active_work) {
            vTaskDelay(pdMS_TO_TICKS(10));
        } else {
            vTaskDelay(1);
        }
    }
}

/* =========================================================================
 * USB C2 Command Parser & Background Task
 * ========================================================================= */

static void parse_and_execute(char *cmd)
{
    cmd[strcspn(cmd, "\r\n")] = 0;
    if (strlen(cmd) == 0) return;

    if (strcmp(cmd, "grant") == 0) {
        if (g_is_hybrid_mode) {
            if (!g_hybrid_permission_granted) {
                g_hybrid_permission_granted = true;
                printf("\n[SYSTEM] >>> HYBRID PERMISSION GRANTED <<<\n");
            } else {
                printf("[SYSTEM] Permission already granted.\n");
            }
        } else {
            printf("[SYSTEM] 'grant' command only applicable in Hybrid Mode (Option 3).\n");
        }
        return;
    }

    if (strcmp(cmd, "stats") == 0) {
        wifi_csi_stats_t stats;
        wifi_csi_get_stats(&stats);
        printf("\n--- Radio HAL Stats ---\n");
        printf("CSI Frames Rx: %lu\n", stats.csi_frames_received);
        printf("Dropped Frames: %lu\n", stats.dropped_frames);
        printf("Scans Executed: %lu\n", stats.scan_count);
        printf("-----------------------\n\n");
        return;
    }

    if (strncmp(cmd, "wifi_connect", 12) == 0) {
        char ssid[33] = {0}, pass[64] = {0};
        int parsed = sscanf(cmd, "wifi_connect \"%32[^\"]\" \"%63[^\"]\"", ssid, pass);
        if (parsed < 2) parsed = sscanf(cmd, "wifi_connect %32s %63s", ssid, pass);

        if (parsed >= 1) {
            printf("[CLI_OK] Connecting to Wi-Fi SSID: '%s'...\n", ssid);
            wifi_config_t wifi_config = {0};
            strncpy((char *)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid) - 1);
            strncpy((char *)wifi_config.sta.password, pass, sizeof(wifi_config.sta.password) - 1);
            esp_wifi_disconnect();
            esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
            esp_wifi_connect();
        } else {
            printf("[CLI_ERR] Usage: wifi_connect \"SSID\" \"PASSWORD\"\n");
        }
        return;
    }

    if (strncmp(cmd, "wifi_prod", 9) == 0) {
        char ip[32] = {0};
        uint16_t port = 3333;
        int parsed = sscanf(cmd, "wifi_prod \"%31[^:]:%hu\"", ip, &port);
        if (parsed < 1) parsed = sscanf(cmd, "wifi_prod %31[^:]:%hu", ip, &port);

        if (parsed >= 1) {
            if (telemetry_transport_set_udp(ip, port) == ESP_OK) {
                printf("[CLI_OK] Output target set to UDP Wireless: %s:%d\n", ip, port);
                printf("[INFO] You can now safely detach the USB cable!\n");
            } else {
                printf("[CLI_ERR] Failed to configure UDP target\n");
            }
        } else {
            printf("[CLI_ERR] Usage: wifi_prod \"192.168.10.16:3333\"\n");
        }
        return;
    }

    if (strcmp(cmd, "usb_mode") == 0) {
        telemetry_transport_set_usb();
        printf("[CLI_OK] Telemetry switched to USB Cable Mode.\n");
        return;
    }

    if (strcmp(cmd, "status") == 0) {
        wifi_ap_record_t ap_info;
        printf("=== SYSTEM STATUS ===\n");
        printf("Transport Mode: %s\n", (telemetry_transport_get_mode() == TELEMETRY_MODE_USB) ? "USB SERIAL" : "UDP WIRELESS");
        if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
            printf("Wi-Fi Status: CONNECTED to '%s' (RSSI: %d dBm)\n", ap_info.ssid, ap_info.rssi);
        } else {
            printf("Wi-Fi Status: DISCONNECTED / PASSIVE\n");
        }
        printf("=====================\n");
        return;
    }

    if (strcmp(cmd, "help") == 0) {
        printf("\n--- Available CLI Commands ---\n");
        printf("1. grant                             - Enable background metadata in Hybrid Mode\n");
        printf("2. stats                             - Print Radio HAL statistics\n");
        printf("3. wifi_connect \"SSID\" \"PASSWORD\"  - Provision and connect Wi-Fi over USB\n");
        printf("4. wifi_prod \"192.168.X.X:PORT\"     - Stream telemetry wirelessly & detach USB\n");
        printf("5. usb_mode                          - Switch telemetry back to USB Cable\n");
        printf("6. status                            - Query network and telemetry state\n");
        printf("------------------------------\n\n");
        return;
    }

    printf("[CLI_ERR] Unknown command: '%s'. Type 'help' for instructions.\n", cmd);
}

static void usb_cli_task(void *pvParameters)
{
    char line_buf[128];
    while (1) {
        if (fgets(line_buf, sizeof(line_buf), stdin) != NULL) {
            parse_and_execute(line_buf);
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

/* =========================================================================
 * Application Entry Point
 * ========================================================================= */

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    printf("==========================================\n");
    printf(" ESP32-S3 Wi-Fi CSI Sensing Orchestrator\n");
    printf("==========================================\n");
    printf("Select Operating Mode:\n");
    printf("[1] Unknown Wi-Fi (Passive Sensing, No Password)\n");
    printf("[2] Known Wi-Fi (Connected, Traditional CSI)\n");
    printf("[3] Known + Unknown (Hybrid, Gated Background Metadata)\n");
    printf("\nEnter choice (1-3): ");

    char choice_buf[8];
    serial_readline(choice_buf, sizeof(choice_buf));
    int mode_choice = atoi(choice_buf);

    app_wifi_csi_config_t hal_config = {0};

    if (mode_choice == 1) {
        printf("\n=> Starting in UNKNOWN (Passive) Mode\n");
        hal_config.mode = WIFI_CSI_MODE_UNKNOWN;
    } 
    else if (mode_choice == 2 || mode_choice == 3) {
        if (mode_choice == 3) {
            printf("\n=> Starting in HYBRID Mode (Requires 'grant' via Serial)\n");
            hal_config.mode = WIFI_CSI_MODE_HYBRID;
            g_is_hybrid_mode = true;
        } else {
            printf("\n=> Starting in KNOWN Mode\n");
            hal_config.mode = WIFI_CSI_MODE_KNOWN;
        }

        char target_ssid[33] = {0};
        char target_pass[65] = {0};
        run_ap_scanner_menu(target_ssid, target_pass);

        strncpy(hal_config.ssid, target_ssid, sizeof(hal_config.ssid));
        strncpy(hal_config.password, target_pass, sizeof(hal_config.password));
    } 
    else {
        printf("\nInvalid choice. Rebooting...\n");
        esp_restart();
    }

    printf("\nInitializing Subsystems...\n");
    
    // Default strictly to USB mode
    telemetry_transport_init("0.0.0.0", 3333);

    // Init Rust Engine (Core 1)
    rust_engine_init();

    // Init Radio HAL
    ESP_ERROR_CHECK(wifi_csi_hal_init(&hal_config));
    ESP_ERROR_CHECK(wifi_csi_hal_start());

    // Spawn Data Router Task pinned to Core 1 (leaving Core 0 free for Wi-Fi stack)
    xTaskCreatePinnedToCore(data_router_task, "data_router", 8192, NULL, 5, NULL, 1);

    // Spawn USB CLI Command Task
    xTaskCreate(usb_cli_task, "usb_cli_task", 4096, NULL, 5, NULL);

    printf("==========================================\n");
    printf(" System Running. Pipeline Active (USB Default).\n");
    printf(" Type 'help' for available USB C2 commands.\n");
    printf("==========================================\n\n");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
