#include <esp_now.h>
#include <WiFi.h>



// Structure example to receive data
// Must match the sender structure
typedef struct struct_message {
    float a;
    float b;
    float c;
    int ecgb;
    
} struct_message;

//FOR SAMPLING START
float CT=0;
float PT=0;
float TD=0;
// FOR SAMPLING END

 int m=0;
 int n=0;



// Create a struct_message called myData
struct_message myData;

// callback function that will be executed when data is received
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  memcpy(&myData, incomingData, sizeof(myData));
  Serial.print(myData.a);
  Serial.print(",");
  Serial.print(myData.b);
  Serial.print(",");
  Serial.print(myData.c);
  Serial.print(",");
  Serial.println(myData.ecgb);
}


void setup() {
  // put your setup code here, to run once:
// Initialize Serial Monitor
  Serial.begin(115200);
  
  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  
  // Once ESPNow is successfully Init, we will register for recv CB to
  // get recv packer info
  esp_now_register_recv_cb(OnDataRecv);
}
 

void loop() {
  // put your main code here, to run repeatedly:

}
