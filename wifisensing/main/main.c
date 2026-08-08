/*
 * main.c
 * System Orchestrator + integrated telemetry (USB/UDP) for ESP32-S3 Wi‑Fi Sensing.
 *
 * This version inlines the telemetry transport and local shared FFI type definitions
 * so the project can build without telemetry_transport.{c,h} or wifi_csi.h.
 *
 * IMPORTANT:
 * - Ensure the struct layouts below exactly match rust_engine/src/types.rs (repr(C)).
 * - Adjust CSI_MAX_SAMPLES if you changed it in Rust.
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_netif.h"
#include "driver/uart.h"
#include "esp_err.h"

static const char *TAG = "ORCHESTRATOR";

/* --------------------------------------------------------------------------
 * Shared constants & FFI types (must match Rust repr(C) layout exactly)
 * ------------------------------------------------------------------------*/
#define CSI_MAX_SAMPLES 384

typedef struct {
    uint64_t timestamp_us;
    uint8_t  mac_address[6];
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  channel;
    uint8_t  bandwidth;
    uint8_t  phy_mode;
    uint16_t csi_length;
    uint8_t  csi_data[CSI_MAX_SAMPLES];
} csi_frame_t;

/* Keep the field order in sync with rust_engine/src/types.rs::MetadataFrame */
typedef struct {
    uint64_t timestamp_us;
    uint8_t  bssid[6];
    char     ssid[33];
    int8_t   rssi;
    uint8_t  channel;
    uint8_t  security_type;
    uint8_t  phy_mode;
    uint8_t  bandwidth;
    int8_t   noise;
    uint16_t beacon_interval;
} metadata_frame_t;

typedef enum {
    WIFI_CSI_MODE_UNKNOWN = 0,
    WIFI_CSI_MODE_KNOWN,
    WIFI_CSI_MODE_HYBRID
} wifi_csi_mode_t;

typedef struct {
    wifi_csi_mode_t mode;
    char ssid[33];
    char password[65];
} app_wifi_csi_config_t;

typedef struct {
    volatile uint32_t csi_frames_received;
    volatile uint32_t dropped_frames;
    volatile uint32_t invalid_frames;
    uint32_t reconnect_count;
    uint32_t disconnect_count;
    uint32_t scan_count;
} wifi_csi_stats_t;

/* --------------------------------------------------------------------------
 * External (remaining) FFI hooks implemented elsewhere in the project
 * - rust_engine_xxx are in the Rust engine (lib with #[no_mangle] symbols)
 * - wifi_csi_hal_xxx and pop APIs are implemented by wifi_csi.c (still present)
 * ------------------------------------------------------------------------*/
extern void rust_engine_init(void);
extern void rust_engine_push_csi(const csi_frame_t *frame);
extern void rust_engine_push_metadata(const metadata_frame_t *meta);

extern esp_err_t wifi_csi_hal_init(const app_wifi_csi_config_t *config);
extern esp_err_t wifi_csi_hal_start(void);
extern bool wifi_csi_pop_frame(csi_frame_t *out_frame, uint32_t timeout_ms);
extern bool wifi_csi_pop_metadata(metadata_frame_t *out_meta, uint32_t timeout_ms);
extern void wifi_csi_get_stats(wifi_csi_stats_t *out_stats);
extern void wifi_csi_reset_stats(void);

/* --------------------------------------------------------------------------
 * Inlined Telemetry Transport (USB framing to stdout + non-blocking UDP)
 * - thread-safe via s_transport_lock
 * - API mirrors prior telemetry_transport.h semantics used by main.c
 * ------------------------------------------------------------------------*/
typedef enum { TELEMETRY_MODE_USB = 0, TELEMETRY_MODE_UDP = 1 } telemetry_mode_t;

static int s_sock = -1;
static struct sockaddr_in s_dest_addr;
static telemetry_mode_t s_active_mode = TELEMETRY_MODE_USB;
static SemaphoreHandle_t s_transport_lock = NULL;

/* Binary sync word for USB framed output */
static const uint8_t SYNC_WORD[4] = { 0xDE, 0xAD, 0xBE, 0xEF };

/* Initialize telemetry subsystem (must be called early) */
static bool telemetry_init(const char *server_ip, uint16_t port)
{
    if (s_transport_lock == NULL) {
        s_transport_lock = xSemaphoreCreateMutex();
        if (s_transport_lock == NULL) {
            ESP_LOGE(TAG, "telemetry_init: failed to create mutex");
            return false;
        }
    }

    /* Disable stdout buffering for immediate USB streaming */
    setvbuf(stdout, NULL, _IONBF, 0);

    if (server_ip && server_ip[0] != '\0' && strcmp(server_ip, "0.0.0.0") != 0) {
        /* attempt UDP */
        // reuse telemetry_set_udp below for consistency
    } else {
        /* remain in USB mode */
    }

    /* start in USB mode by default; optionally configure UDP below */
    s_active_mode = TELEMETRY_MODE_USB;
    s_sock = -1;
    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    return true;
}

static esp_err_t telemetry_set_udp(const char *ip_str, uint16_t port)
{
    if (!ip_str || port == 0) return ESP_ERR_INVALID_ARG;
    if (!s_transport_lock) return ESP_ERR_INVALID_STATE;

    xSemaphoreTake(s_transport_lock, portMAX_DELAY);

    /* Close previous socket if any */
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }

    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    s_dest_addr.sin_family = AF_INET;
    s_dest_addr.sin_port = htons(port);

    int pton = inet_pton(AF_INET, ip_str, &s_dest_addr.sin_addr);
    if (pton != 1) {
        ESP_LOGE(TAG, "telemetry_set_udp: invalid IP '%s'", ip_str);
        xSemaphoreGive(s_transport_lock);
        return ESP_ERR_INVALID_ARG;
    }

    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "telemetry_set_udp: socket() failed errno=%d", errno);
        s_active_mode = TELEMETRY_MODE_USB;
        xSemaphoreGive(s_transport_lock);
        return ESP_FAIL;
    }

    int flags = fcntl(s_sock, F_GETFL, 0);
    if (flags >= 0) {
        if (fcntl(s_sock, F_SETFL, flags | O_NONBLOCK) < 0) {
            ESP_LOGW(TAG, "telemetry_set_udp: fcntl set O_NONBLOCK failed (errno=%d)", errno);
            /* not fatal, continue */
        }
    }

    s_active_mode = TELEMETRY_MODE_UDP;
    ESP_LOGI(TAG, "Telemetry -> UDP [%s:%d]", ip_str, port);

    xSemaphoreGive(s_transport_lock);
    return ESP_OK;
}

static void telemetry_set_usb(void)
{
    if (!s_transport_lock) return;
    xSemaphoreTake(s_transport_lock, portMAX_DELAY);

    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }
    s_active_mode = TELEMETRY_MODE_USB;
    ESP_LOGI(TAG, "Telemetry -> USB (stdout)");

    xSemaphoreGive(s_transport_lock);
}

static telemetry_mode_t telemetry_get_mode(void)
{
    telemetry_mode_t mode = TELEMETRY_MODE_USB;
    if (!s_transport_lock) return mode;
    xSemaphoreTake(s_transport_lock, portMAX_DELAY);
    mode = s_active_mode;
    xSemaphoreGive(s_transport_lock);
    return mode;
}

/* Send telemetry payload. Returns bytes written, 0 if dropped, -1 on error. */
static int telemetry_send(const void *payload, size_t size)
{
    if (!payload || size == 0) return -1;
    if (!s_transport_lock) return -1;

    xSemaphoreTake(s_transport_lock, portMAX_DELAY);

    if (s_active_mode == TELEMETRY_MODE_USB) {
        /* Frame to stdout: [SYNC_WORD(4)][len(2)][payload] */
        size_t nw = fwrite(SYNC_WORD, 1, sizeof(SYNC_WORD), stdout);
        (void)nw;
        uint16_t frame_size = (uint16_t)size;
        fwrite(&frame_size, 1, sizeof(frame_size), stdout);
        size_t written = fwrite(payload, 1, size, stdout);
        fflush(stdout);
        xSemaphoreGive(s_transport_lock);
        return (int)written;
    }

    if (s_active_mode == TELEMETRY_MODE_UDP && s_sock >= 0) {
        int bytes_sent = sendto(s_sock, payload, size, 0, (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
        if (bytes_sent > 0) {
            xSemaphoreGive(s_transport_lock);
            return bytes_sent;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            /* drop silently to avoid blocking pipeline */
            xSemaphoreGive(s_transport_lock);
            return 0;
        }
        ESP_LOGW(TAG, "telemetry_send: sendto failed errno=%d", errno);
    }

    xSemaphoreGive(s_transport_lock);
    return -1;
}

static void telemetry_deinit(void)
{
    if (!s_transport_lock) return;
    xSemaphoreTake(s_transport_lock, portMAX_DELAY);
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }
    s_active_mode = TELEMETRY_MODE_USB;
    xSemaphoreGive(s_transport_lock);
}

/* --------------------------------------------------------------------------
 * Orchestrator: main app logic (largely unchanged)
 * ------------------------------------------------------------------------*/
static volatile bool g_is_hybrid_mode = false;
static volatile bool g_hybrid_permission_granted = false;

/* Serial readline helper (non-blocking-ish) */
static void serial_readline(char *buf, size_t max_len)
{
    size_t count = 0;
    while (count < max_len - 1) {
        int c = getchar();
        if (c == '\n' || c == '\r') {
            if (count > 0) break;
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

/* Local AP scanner menu; uses esp_wifi APIs */
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
        } else {
            /* allocation failed: fallback to empty */
            out_ssid[0] = '\0';
            out_password[0] = '\0';
            ESP_LOGW(TAG, "run_ap_scanner_menu: malloc failed, using empty SSID");
        }
    }

    ESP_ERROR_CHECK(esp_wifi_stop());
    ESP_ERROR_CHECK(esp_wifi_deinit());
    if (temp_netif) {
        esp_netif_destroy_default_wifi(temp_netif);
    }

    printf("--- Scanner Teardown Complete ---\n\n");
}

/* Data router task (pins to Core 1 in app_main) */
static void data_router_task(void *pvParameters)
{
    static csi_frame_t csi;
    static metadata_frame_t meta;

    uint32_t pop_count = 0;
    uint32_t send_count = 0;
    TickType_t last_log = xTaskGetTickCount();

    while (1) {
        bool active_work = false;

        if (wifi_csi_pop_frame(&csi, 10)) {
            rust_engine_push_csi(&csi);
            telemetry_send(&csi, sizeof(csi));
            active_work = true;
            pop_count++;
            send_count++;
        }

        if (wifi_csi_pop_metadata(&meta, 10)) {
            if (!g_is_hybrid_mode || g_hybrid_permission_granted) {
                rust_engine_push_metadata(&meta);
            }
            active_work = true;
        }

        if ((xTaskGetTickCount() - last_log) > pdMS_TO_TICKS(5000)) {
            wifi_csi_stats_t stats_snapshot;
            memset(&stats_snapshot, 0, sizeof(stats_snapshot));
            wifi_csi_get_stats(&stats_snapshot);

            ESP_LOGI("DATA_ROUTER", "popped=%u sent=%u dropped=%u",
                     (unsigned)pop_count,
                     (unsigned)send_count,
                     (unsigned)stats_snapshot.dropped_frames);

            last_log = xTaskGetTickCount();
        }

        if (!active_work) {
            vTaskDelay(pdMS_TO_TICKS(10));
        } else {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

/* CLI command parser */
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
            if (telemetry_set_udp(ip, port) == ESP_OK) {
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
        telemetry_set_usb();
        printf("[CLI_OK] Telemetry switched to USB Cable Mode.\n");
        return;
    }

    if (strcmp(cmd, "status") == 0) {
        wifi_ap_record_t ap_info;
        printf("=== SYSTEM STATUS ===\n");
        printf("Transport Mode: %s\n", (telemetry_get_mode() == TELEMETRY_MODE_USB) ? "USB SERIAL" : "UDP WIRELESS");
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

/* Application entrypoint */
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
    } else if (mode_choice == 2 || mode_choice == 3) {
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
    } else {
        printf("\nInvalid choice. Rebooting...\n");
        esp_restart();
    }

    printf("\nInitializing Subsystems...\n");

    /* Initialize telemetry to USB by default (0.0.0.0 -> USB) */
    telemetry_init("0.0.0.0", 3333);

    /* Initialize Rust engine (no-op if not implemented) */
    rust_engine_init();

    /* Initialize Radio HAL (wifi_csi.c) */
    ESP_ERROR_CHECK(wifi_csi_hal_init(&hal_config));
    ESP_ERROR_CHECK(wifi_csi_hal_start());

    /* Spawn Data Router pinned to core 1 */
    xTaskCreatePinnedToCore(data_router_task, "data_router", 8192, NULL, 5, NULL, 1);

    /* Spawn USB CLI Task */
    xTaskCreate(usb_cli_task, "usb_cli_task", 4096, NULL, 5, NULL);

    printf("==========================================\n");
    printf(" System Running. Pipeline Active (USB Default).\n");
    printf(" Type 'help' for available USB C2 commands.\n");
    printf("==========================================\n\n");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
