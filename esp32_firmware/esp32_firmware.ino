#define CAMERA_MODEL_ESP32S3_EYE

#if defined(CAMERA_MODEL_AI_THINKER)
#define PIN_MOTOR_A_PWM 12
#define PIN

_MOTOR_B_PWM 13
#elif defined(CAMERA_MODEL_ESP32S3_EYE)
#define PIN_MOTOR_A_PWM 42  // Đã đảo lại chân do đấu nối ngược
#define PIN_MOTOR_B_PWM 41
#endif

#define DEBUG_MODE

#include "camera_pins.h"
#include "driver/uart.h"
#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"      
#include <WiFi.h>
#include <WiFiUdp.h>
#include <cstring>
#include <stdio.h>

// --- WIFI CONFIGURATION ---
const char *ssid = "YOUR_WIFI_SSID";        // Đổi thành tên Wi-Fi thật trước khi nạp
const char *password = "YOUR_WIFI_PASSWORD"; // Đổi thành mật khẩu Wi-Fi thật
const char *udpAddressLaptop = "192.168.137.185"; // IP máy tính của bạn (Sẽ được tự động cập nhật khi PC chạy script thu data)
const uint16_t udpPortLaptop = 3000;    // Port nhận ảnh trên PC
const uint16_t udpPortCam = 3001;       // Port nhận lệnh trên ESP32

WiFiUDP Udp;

// --- DC MOTOR CONTROL CLASS ---
#define pwmFrequency 800 // Tần số PWM cho động cơ
#define pwmResolution 8  // Độ phân giải 8-bit (0-255)

#define STRAIGHT 1
#define SLOW 2
#define LEFT 3
#define RIGHT 4
#define TURN_LEFT 5
#define TURN_RIGHT 6
#define STANDBY 7

class DCMotorControl {
private:
    uint8_t pinPWNMotorA;
    uint8_t pinPWNMotorB;

public:
    DCMotorControl(uint8_t pinPWNMotorA, uint8_t pinPWNMotorB) {
        this->pinPWNMotorA = pinPWNMotorA;
        this->pinPWNMotorB = pinPWNMotorB;

        ledcAttachChannel(pinPWNMotorA, pwmFrequency, pwmResolution, 2);
        ledcAttachChannel(pinPWNMotorB, pwmFrequency, pwmResolution, 3);
    }

    void SettingMotor(uint8_t speedMotorA, uint8_t speedMotorB) {
        ledcWrite(pinPWNMotorA, speedMotorA);
        ledcWrite(pinPWNMotorB, speedMotorB);
    }

    void CarMovementControl(uint8_t direction, uint8_t speed, int8_t alpha) {
        switch (direction) {
        case STRAIGHT:
            SettingMotor(speed, speed - alpha);
            break;
        case SLOW:
            SettingMotor(speed - alpha, speed - alpha);
            break;
        case LEFT:
            SettingMotor(speed - alpha, speed);
            break;
        case RIGHT:
            SettingMotor(speed, speed - alpha);
            break;
        case TURN_LEFT:
            SettingMotor(0, 0);
            delay(100);
            SettingMotor(speed, speed);
            delay(500);
            SettingMotor(0, speed);
            delay(alpha * 100);
            break;
        case TURN_RIGHT:
            SettingMotor(0, 0);
            delay(100);
            SettingMotor(speed, speed);
            delay(500);
            SettingMotor(speed, 0);
            delay(alpha * 100);
            break;
        case STANDBY:
            SettingMotor(speed, speed);
            delay(alpha * 100);
            break;
        default:
            SettingMotor(0, 0); // Dừng xe
            break;
        }
    }
};

// --- CAMERA CONFIGURATION ---
static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 10,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
};

#if defined(DEBUG_MODE)
void uart_init() {
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE};
    uart_param_config(UART_NUM_1, &uart_config);
    uart_set_pin(UART_NUM_1, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM_1, 256, 0, 0, NULL, 0);
}
#endif

esp_err_t camera_init() {
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        printf("Camera Init Failed\n");
        return err;
    }

    sensor_t *pSensor = esp_camera_sensor_get();
    pSensor->set_vflip(pSensor, 0);   // Có lật ngược ảnh không
    pSensor->set_hmirror(pSensor, 0); // Có gương lật ngang không

    printf("Camera Init OK\n");
    return ESP_OK;
}

void wifi_init() {
    WiFi.begin(ssid, password);
    Serial.println("");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    printf("\nConnected to WiFi\n");
    Udp.begin(udpPortCam);
    printf("Now listening at IP %s port %d\n", WiFi.localIP().toString().c_str(), udpPortCam);
}

// --- FREE RTOS TASKS ---
IPAddress pcIP;
bool pcConnected = false;

void sendImageTask(void *pvParameters) {
    camera_fb_t *fb = NULL;
    printf("Send image task started\n");
    while (true) {
        // Chỉ gửi ảnh khi máy tính đã gửi tín hiệu điều khiển đầu tiên (để lấy địa chỉ IP của PC tự động)
        if (pcConnected) {
            fb = esp_camera_fb_get();
            if (!fb) {
                printf("Camera Capture Failed\n");
                delay(100);
                continue;
            }

            Udp.beginPacket(pcIP, udpPortLaptop);
            Udp.write(fb->buf, fb->len);
            Udp.endPacket();
            esp_camera_fb_return(fb);
        } else {
            // Nếu chưa có kết nối, thử gửi đến IP mặc định
            fb = esp_camera_fb_get();
            if (fb) {
                Udp.beginPacket(udpAddressLaptop, udpPortLaptop);
                Udp.write(fb->buf, fb->len);
                Udp.endPacket();
                esp_camera_fb_return(fb);
            }
            delay(500); // Gửi chậm khi chưa kết nối
        }
        vTaskDelay(pdMS_TO_TICKS(30)); // Giới hạn tần số khoảng ~30 FPS
    }
}

void receiveMessageTask(void *pvParameters) {
    DCMotorControl motorControl(PIN_MOTOR_A_PWM, PIN_MOTOR_B_PWM);
    uint8_t packetSize;
    char MovementCmd[15];
    uint8_t directionLength;
    uint8_t speedLength;
    uint8_t alphaLength;

    char *directionStr;
    char *speedStr;
    char *alphaStr;
    char *spacePos1;
    char *spacePos2;

    uint8_t unDirection;
    uint8_t unSpeed;
    int8_t nAlpha;
    uint8_t len;

    printf("Listen task started\n");
    while (true) {
        packetSize = Udp.parsePacket();
        if (packetSize) {
            // Lấy IP của máy tính gửi lệnh tự động
            pcIP = Udp.remoteIP();
            pcConnected = true;

            len = Udp.read(MovementCmd, 15);
            if (len > 0) {
                MovementCmd[len] = '\0';
                
                // Silently handle heartbeat packets for IP discovery
                if (strcmp(MovementCmd, "HB") != 0 && strcmp(MovementCmd, "HB\n") != 0) {
                    printf("Command received: %s\n", MovementCmd);

                    spacePos1 = strchr(MovementCmd, ' ');
                    spacePos2 = nullptr;
                    if (spacePos1 != nullptr) {
                        spacePos2 = strchr(spacePos1 + 1, ' ');
                    }

                    if (spacePos1 != nullptr && spacePos2 != nullptr) {
                        directionLength = spacePos1 - MovementCmd;
                        speedLength = spacePos2 - (spacePos1 + 1);
                        alphaLength = len - (spacePos2 - MovementCmd) - 1;

                        directionStr = MovementCmd;
                        speedStr = spacePos1 + 1;
                        alphaStr = spacePos2 + 1;

                        unDirection = atoi(directionStr);
                        unSpeed = atoi(speedStr);
                        nAlpha = atoi(alphaStr);

                        printf("Parsed -> Dir: %d, Speed: %d, Alpha: %d\n", unDirection, unSpeed, nAlpha);

                        motorControl.CarMovementControl(unDirection, unSpeed, nAlpha);
                    } else {
                        printf("Invalid command format\n");
                    }
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

void setup() {
    Serial.begin(115200);
    uart_init();
    esp_err_t cam_err = camera_init();

if (cam_err != ESP_OK) {
    Serial.printf("CAMERA INIT ERROR: 0x%x\n", cam_err);

    while (true) {
        delay(1000);
    }
}

Serial.println("CAMERA READY");
wifi_init();

    // Khởi tạo các task chạy đa nhân độc lập trên FreeRTOS
    xTaskCreatePinnedToCore(
        sendImageTask,
        "SendImageTask",
        8192,
        NULL,
        2,
        NULL,
        1); // Camera & Send: chạy trên Core 1

    xTaskCreatePinnedToCore(
        receiveMessageTask,
        "RecvMsgTask",
        4096,
        NULL,
        1,
        NULL,
        0); // Motor & UDP Control: chạy trên Core 0
}

void loop() {
    // Để trống vì FreeRTOS quản lý các Task nhúng
}
