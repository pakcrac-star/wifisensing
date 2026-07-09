/* name=main.c
 *
 * Robust boot UX and runtime orchestration for ESP32-S3 WiFi CSI Radar.
 * - Boot selection (PASSIVE / ACTIVE)
 * - CSI init with graceful fallback to Wi-Fi-only scanning
 * - ACTIVE: scan + prompt + connect (waits for IP)
 * - PASSIVE: uses CSI frames for inference; otherwise runs periodic scans and emits RSSI-based telemetry
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_timer.h"

#include "wifi_csi.h"
#include "model.h"
#include "json_sender.h"
#include "telemetry_transport.h" /* your added transport implementation (prints to UART or custom) */

#define TAG "CSI_RADAR"

#define BOOT_PROMPT_TIMEOUT_MS    20000
#define ACTIVE_CONNECT_TIMEOUT_MS 10000
#define PASSIVE_SCAN_INTERVAL_MS  5000

typedef enum {
    BOOT_MODE_UNSET = 0,
    BOOT_MODE_PASSIVE,
    BOOT_MODE_ACTIVE
} boot_mode_t;

static volatile boot_mode_t g_boot_mode = BOOT_MODE_UNSET;

/* runtime flags */
static bool g_csi_driver_available = false;
static bool g_csi_running = false;
static bool g_wifi_driver_available = false;

/* TinyML context */
static tinyml_ctx_t static_ml_ctx;

/* Wi-Fi event group */
static EventGroupHandle_t s_wifi_event_group;
const int CONNECTED_BIT = BIT0;

/* Safe UART line reader */
static void safe_get_string(char *buffer, size_t max_len)
{
    size_t idx = 0;
    while (idx < max_len - 1) {
        int c = getchar();
        if (c == EOF || c == '\n' || c == '\r') {
            if (idx > 0) break;
            continue;
        }
        buffer[idx++] = (char)c;
        putchar(c);
    }
    buffer[idx] = '\0';
    printf("\n");
}

/* Boot prompt task */
static void boot_prompt_task(void *arg)
{
    (void)arg;
    printf("\nBoot menu active:\n");
    printf("  [1] Wi-Fi WITHOUT known password (passive CSI + RSSI monitoring)\n");
    printf("  [2] Wi-Fi WITH password (scan, connect, CSI + connected telemetry)\n");
    printf("Enter selection (1 or 2) within %d s to override default (passive):\n", BOOT_PROMPT_TIMEOUT_MS / 1000);

    char choice = 0;
    const TickType_t timeout = pdMS_TO_TICKS(BOOT_PROMPT_TIMEOUT_MS);
    TickType_t start = xTaskGetTickCount();

    while ((xTaskGetTickCount() - start) < timeout) {
        int c = getchar();
        if (c != EOF && c != '\n' && c != '\r') {
            choice = (char)c;
            putchar(choice);
            printf("\n");
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    if (choice == '2') {
        g_boot_mode = BOOT_MODE_ACTIVE;
        printf("Selected: ACTIVE mode (scan & STA connect).\n");
    } else if (choice == '1') {
        g_boot_mode = BOOT_MODE_PASSIVE;
        printf("Selected: PASSIVE mode (CSI + RSSI only).\n");
    } else {
        g_boot_mode = BOOT_MODE_PASSIVE;
        printf("No selection: defaulting to PASSIVE mode.\n");
    }

    vTaskDelete(NULL);
}

/* Wi-Fi / IP event handler */
static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)
{
    (void)arg;
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                ESP_LOGI(TAG, "WIFI_EVENT_STA_START");
                break;
            case WIFI_EVENT_STA_CONNECTED:
                ESP_LOGI(TAG, "WIFI_EVENT_STA_CONNECTED");
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                ESP_LOGW(TAG, "WIFI_EVENT_STA_DISCONNECTED");
                xEventGroupClearBits(s_wifi_event_group, CONNECTED_BIT);
                break;
            default:
                break;
        }
    } else if (event_base == IP_EVENT) {
        if (event_id == IP_EVENT_STA_GOT_IP) {
            ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
            char ipbuf[16] = {0};
            esp_ip4addr_ntoa(&event->ip_info.ip, ipbuf, sizeof(ipbuf));
            ESP_LOGI(TAG, "Got IP: %s", ipbuf);
            xEventGroupSetBits(s_wifi_event_group, CONNECTED_BIT);
        }
    }
}

/* Fallback Wi-Fi init (no CSI) */
static esp_err_t wifi_init_fallback_for_scan(void)
{
    esp_err_t ret;
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

    ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Fallback esp_wifi_init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ret = esp_wifi_set_mode(WIFI_MODE_STA);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Fallback esp_wifi_set_mode failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_wifi_start();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Fallback esp_wifi_start failed: %s", esp_err_to_name(ret));
        return ret;
    }

    return ESP_OK;
}

/* Simple RSSI->confidence mapping */
static float rssi_to_confidence(int rssi)
{
    if (rssi <= -100) return 0.0f;
    if (rssi >= -30)  return 1.0f;
    return (float)(rssi + 100) / 70.0f;
}

/* Blocking scan -> best RSSI */
static int perform_blocking_scan_best_rssi(void)
{
    uint16_t max_ap = 32;
    wifi_ap_record_t ap_records[32];
    esp_err_t rc = esp_wifi_scan_start(NULL, true);
    if (rc != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_scan_start failed: %s", esp_err_to_name(rc));
        return INT32_MIN;
    }
    uint16_t found = 0;
    rc = esp_wifi_scan_get_ap_records(&max_ap, ap_records);
    if (rc != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_scan_get_ap_records failed: %s", esp_err_to_name(rc));
        return INT32_MIN;
    }
    rc = esp_wifi_scan_get_ap_num(&found);
    if (rc != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_scan_get_ap_num failed: %s", esp_err_to_name(rc));
        return INT32_MIN;
    }
    if (found == 0) return INT32_MIN;

    int best = INT32_MIN;
    for (uint16_t i = 0; i < found && i < max_ap; ++i) {
        if (ap_records[i].rssi > best) best = ap_records[i].rssi;
    }
    return best;
}

/* CSI processing worker */
static void csi_processing_task(void *pvParameters)
{
    (void)pvParameters;
    CSIFrame frame;
    CSIFeatures features;
    TinyMLResult result;

    ESP_LOGI(TAG, "CSI worker started (CSI_running=%d, WiFi_available=%d)", (int)g_csi_running, (int)g_wifi_driver_available);

    while (1) {
        if (g_csi_running && wifi_csi_available()) {
            if (wifi_csi_read(&frame)) {
                if (tinyml_extract_features(&static_ml_ctx, &frame, &features) != ESP_OK) {
                    ESP_LOGW(TAG, "tinyml_extract_features failed");
                    continue;
                }
                if (tinyml_predict(&features, &result) == ESP_OK) {
                    bool active = (g_boot_mode == BOOT_MODE_ACTIVE);
                    json_sender_send(&features, &result, &frame, active);
                }
            }
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        /* Fallback: no CSI. If Wi-Fi available and PASSIVE mode, perform scan-based telemetry */
        if (!g_csi_running && g_wifi_driver_available && g_boot_mode == BOOT_MODE_PASSIVE) {
            int best_rssi = perform_blocking_scan_best_rssi();
            if (best_rssi != INT32_MIN) {
                memset(&features, 0, sizeof(features));
                features.amplitude_mean = (float)best_rssi;
                features.snr_estimate = (float)best_rssi;

                result.motion_detected = false;
                result.person_present = (best_rssi > -70);
                result.motion_score = (float)(-best_rssi);
                result.confidence = rssi_to_confidence(best_rssi);
                result.timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);

                CSIFrame fake_frame;
                memset(&fake_frame, 0, sizeof(fake_frame));
                fake_frame.rssi = (int8_t)best_rssi;
                fake_frame.timestamp = result.timestamp_ms;

                json_sender_send(&features, &result, &fake_frame, false);
            } else {
                ESP_LOGW(TAG, "Passive fallback scan returned no APs or failed.");
            }
            vTaskDelay(pdMS_TO_TICKS(PASSIVE_SCAN_INTERVAL_MS));
            continue;
        }

        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

/* Application entry */
void app_main(void)
{
    esp_err_t ret;

    /* NVS */
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    s_wifi_event_group = xEventGroupCreate();
    if (s_wifi_event_group == NULL) {
        ESP_LOGW(TAG, "Failed to create Wi-Fi event group; continuing without connection wait capability.");
    }

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    printf("\n================================================\n");
    printf("       ESP32-S3 WiFi CSI Radar Boot Menu        \n");
    printf("================================================\n\n");
    printf("Starting boot selection. Default is PASSIVE mode.\n");

    xTaskCreate(boot_prompt_task, "boot_prompt", 4096, NULL, 2, NULL);

    const TickType_t wait_timeout = pdMS_TO_TICKS(BOOT_PROMPT_TIMEOUT_MS + 1000);
    TickType_t start = xTaskGetTickCount();
    while (g_boot_mode == BOOT_MODE_UNSET && (xTaskGetTickCount() - start) < wait_timeout) {
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    if (g_boot_mode == BOOT_MODE_UNSET) g_boot_mode = BOOT_MODE_PASSIVE;

    ESP_LOGI(TAG, "Initializing TinyML context...");
    ret = tinyml_init(&static_ml_ctx);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "tinyml_init failed: %s", esp_err_to_name(ret));
    }

    ESP_LOGI(TAG, "Attempting to initialize CSI engine...");
    ret = wifi_csi_init();
    if (ret == ESP_OK) {
        g_csi_driver_available = true;
        g_wifi_driver_available = true;
        ESP_LOGI(TAG, "wifi_csi_init succeeded (CSI driver available).");
    } else {
        ESP_LOGW(TAG, "wifi_csi_init failed: %s. Trying fallback Wi-Fi init for scans/connects.", esp_err_to_name(ret));
        ret = wifi_init_fallback_for_scan();
        if (ret == ESP_OK) {
            g_wifi_driver_available = true;
            g_csi_driver_available = false;
            ESP_LOGI(TAG, "Fallback Wi-Fi init succeeded (CSI not available).");
        } else {
            g_wifi_driver_available = false;
            g_csi_driver_available = false;
            ESP_LOGW(TAG, "No Wi-Fi driver available. Running in offline/degraded mode.");
        }
    }

    if (g_csi_driver_available) {
        ret = wifi_csi_start();
        if (ret == ESP_OK) {
            g_csi_running = true;
            ESP_LOGI(TAG, "CSI acquisition engine started.");
        } else {
            g_csi_running = false;
            ESP_LOGW(TAG, "wifi_csi_start failed: %s. Continuing without CSI running.", esp_err_to_name(ret));
        }
    }

    if (g_boot_mode == BOOT_MODE_ACTIVE) {
        if (!g_wifi_driver_available) {
            ESP_LOGW(TAG, "ACTIVE mode selected but no Wi-Fi driver available; falling back to PASSIVE behavior.");
            g_boot_mode = BOOT_MODE_PASSIVE;
        } else {
            printf("ACTIVE mode: scanning nearby networks (blocking)...\n");
            uint16_t max_ap = 20;
            wifi_ap_record_t ap_info[20];
            uint16_t ap_count = 0;
            esp_err_t rc = esp_wifi_scan_start(NULL, true);
            if (rc != ESP_OK) {
                ESP_LOGW(TAG, "Scan start failed: %s", esp_err_to_name(rc));
            } else {
                rc = esp_wifi_scan_get_ap_records(&max_ap, ap_info);
                if (rc != ESP_OK) {
                    ESP_LOGW(TAG, "Scan get records failed: %s", esp_err_to_name(rc));
                } else {
                    rc = esp_wifi_scan_get_ap_num(&ap_count);
                    if (rc != ESP_OK) {
                        ESP_LOGW(TAG, "Scan get num failed: %s", esp_err_to_name(rc));
                    } else {
                        printf("\n--- Found %d Available Networks ---\n", ap_count);
                        for (int i = 0; i < (int)max_ap && i < (int)ap_count; ++i) {
                            printf("[%d] SSID: %s (RSSI: %d)\n", i + 1, ap_info[i].ssid, ap_info[i].rssi);
                        }
                    }
                }
            }

            char target_ssid[33] = {0};
            char target_password[65] = {0};

            printf("\nEnter exact Target SSID: ");
            safe_get_string(target_ssid, sizeof(target_ssid));

            printf("Enter Password: ");
            safe_get_string(target_password, sizeof(target_password));

            wifi_config_t wifi_config = {0};
            strncpy((char *)wifi_config.sta.ssid, target_ssid, sizeof(wifi_config.sta.ssid) - 1);
            strncpy((char *)wifi_config.sta.password, target_password, sizeof(wifi_config.sta.password) - 1);

            esp_err_t rc2 = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
            if (rc2 != ESP_OK) {
                ESP_LOGW(TAG, "esp_wifi_set_config failed: %s", esp_err_to_name(rc2));
            } else {
                ESP_LOGI(TAG, "Connecting to '%s'...", target_ssid);
                rc2 = esp_wifi_connect();
                if (rc2 != ESP_OK) {
                    ESP_LOGW(TAG, "esp_wifi_connect failed: %s", esp_err_to_name(rc2));
                } else {
                    if (s_wifi_event_group != NULL) {
                        EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group, CONNECTED_BIT, pdFALSE, pdFALSE, pdMS_TO_TICKS(ACTIVE_CONNECT_TIMEOUT_MS));
                        if (bits & CONNECTED_BIT) {
                            ESP_LOGI(TAG, "Connected to AP and got IP.");
                        } else {
                            ESP_LOGW(TAG, "Connection attempt timed out.");
                        }
                    } else {
                        vTaskDelay(pdMS_TO_TICKS(ACTIVE_CONNECT_TIMEOUT_MS));
                        ESP_LOGI(TAG, "Active connect wait finished (no event group).");
                    }
                }
            }
        }
    } else {
        if (g_csi_running) {
            ESP_LOGI(TAG, "PASSIVE mode with CSI running.");
        } else if (g_wifi_driver_available) {
            ESP_LOGI(TAG, "PASSIVE mode without CSI: Wi-Fi driver available for periodic scans.");
        } else {
            ESP_LOGW(TAG, "PASSIVE mode: no CSI and no Wi-Fi driver available.");
        }
    }

    xTaskCreate(csi_processing_task, "csi_proc", 8192, NULL, 5, NULL);

    ESP_LOGI(TAG, "app_main completed initialization.");
}