/////*********ESP-NOW CODE*********
#include <esp_now.h>
#include <WiFi.h>
uint8_t broadcastAddress[] = {0x7C, 0xDF, 0xA1, 0x55, 0x2F, 0x9E};
// Structure example to send data
// Must match the receiver structure
typedef struct struct_message {
  float a;
  float b;
  float c;
  int d;
} struct_message;

// Create a struct_message called myData
struct_message myData;
esp_now_peer_info_t peerInfo;
// callback when data is sent
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("\r\nLast Packet Send Status:\t");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}


//*********SENSOR CODE*********



//sensor pin call
int SAPL=8;
int SAPM=4;
int SAPR=6;
//known value call
float R2 = 2200;  //2.2K known resistor
int Vin = 3;
int bits=8191; // 12 bit ADC 

//unknown variable call

float VoutL = 0;
float VoutM = 0;
float VoutR = 0;

float RL = 0; // sensor left
float RM = 0; // sensor Middle
float RR = 0; // sensor Right
float EQL1=0;
float EQM1=0;
float EQR1=0;
float EQL2=0;
float EQM2=0;
float EQR2=0;
float rawL=0;
float rawM=0;
float rawR=0;

float CT=0;
float PT=0;
float TD=0;

 int m=0;
 int n=0;




 /////ECG START

#include<SPI.h>
#include "protocentral_max30003.h"

#define MAX30003_CS_PIN A5

MAX30003 max30003(MAX30003_CS_PIN);


 /////ECG END

 

//****************************************************************************************************************
//****************************************************************************************************************
void setup(){
  
//*********SENSOR CODE*********
pinMode(SAPL,INPUT);
pinMode(SAPM,INPUT);
pinMode(SAPR,INPUT);
Serial.begin(115200);

/////*********ESP-NOW CODE*********
// Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);
    // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
    }
  // Once ESPNow is successfully Init, we will register for Send CB to
  // get the status of Trasnmitted packet
  esp_now_register_send_cb(OnDataSent);
  // Register peer
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  // Add peer        
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
    }


 /////ECG START


pinMode(MAX30003_CS_PIN,OUTPUT);
    digitalWrite(MAX30003_CS_PIN,HIGH); //disable device

    SPI.begin();

    bool ret = max30003.max30003ReadInfo();
    if(ret){
      Serial.println("Max30003 read ID Success");
    }else{

      while(!ret){
        //stay here untill the issue is fixed.
        ret = max30003.max30003ReadInfo();
        Serial.println("Failed to read ID, please make sure all the pins are connected");
        delay(10000);
      }
    }

    Serial.println("Initialising the chip ...");
    max30003.max30003Begin();   // initialize MAX30003


 /////ECG END




    
}

//****************************************************************************************************************
//****************************************************************************************************************

void loop(){

  //Sensor Left Pin 4, (red)
  rawL = analogRead(SAPL);
  if(rawL){
    EQL1 = rawL * Vin;
    VoutL = (EQL1)/bits;
    EQL2 = (Vin/VoutL) - 1;
    RL= (R2 * EQL2) +100;

  }
 //Sensor Left Pin 6, (black)

  rawM = analogRead(SAPM);
  if(rawM){
    EQM1 = rawM * Vin;
    VoutM = (EQM1)/bits;
    EQM2 = (Vin/VoutM) - 1;
    RM= (R2 * EQM2)-700;


  }

  //Sensor Left Pin 8, (white)

  rawR = analogRead(SAPR);
  if(rawR){
    EQR1 = rawR * Vin;
    VoutR = (EQR1)/bits;
    EQR2 = (Vin/VoutR) - 1;
    RR= (R2 * EQR2)-300;
  
  }
    
    
    Serial.print(RL);
    Serial.print(",");
    Serial.print(RM);
    Serial.print(",");
    Serial.println(RR);
    // 100Hz Sample rate now.

// Chekcing Sample Rate
    CT=millis();
    TD=CT-PT; // for the time it takes to complete each loops
    PT=CT;
    n=m+1; // to count the numbe of data printed over the that time
    Serial.print(n);
    Serial.print(",");
    Serial.println(TD);
    m=n;
    




 /////ECG START


    max30003.getEcgSamples();   //It reads the ecg sample and stores it to max30003.ecgdata .
    int ecga = max30003.ecgdata;
    int ecgb = map (ecga, 0,-1500000, 0,1023);
    
    //Serial.println(ecgb);


  /////ECG END
 


//************************ESP-NOW*******************************

 // Set values to send
 
  strcpy;
  myData.a = RL;
  myData.b = RM;
  myData.c = RR;
  myData.d = ecgb;
  
// Send message via ESP-NOW
  esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
   
  if (result == ESP_OK) {
    Serial.println("Sent with success");
  }
  else {
    Serial.println("Error sending the data");
  }
   delay (9);
}
