#!/bin/bash

# --- CONFIGURATION ---
# Default paths
# Input: Merged data folder
# Output: Separate chunked data folder
IN_BASE_DIR=${1:-"/Volumes/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/4_merged_lables/PolarH10/PolarH10"}
OUT_BASE_DIR=${2:-"/Volumes/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/5_activity_chunks/PolarH10"}

# Check if input directory exists
if [ ! -d "$IN_BASE_DIR" ]; then
    echo "❌ Error: PolarH10 merged data directory not found: $IN_BASE_DIR"
    echo "Usage: ./run_chunk_polarh10.sh [IN_BASE_DIR] [OUT_BASE_DIR]"
    exit 1
fi

echo "🚀 Starting PolarH10 Chunking Pipeline..."
echo "📂 Input Base:  $IN_BASE_DIR"
echo "📂 Output Base: $OUT_BASE_DIR"
echo "------------------------------------------"

# List of sensors to process
SENSORS=("ACC" "HR" "ECG" "IBI")

for SENSOR in "${SENSORS[@]}"; do
    SENSOR_IN_DIR="$IN_BASE_DIR/$SENSOR"
    SENSOR_OUT_DIR="$OUT_BASE_DIR/$SENSOR"
    
    if [ -d "$SENSOR_IN_DIR" ]; then
        echo "📝 Chunking Sensor: $SENSOR"
        
        # Run python script with explicit arguments
        python3 PolarH10_chunking.py --input "$SENSOR_IN_DIR" --output "$SENSOR_OUT_DIR" --sensor "$SENSOR"
        
        if [ $? -eq 0 ]; then
            echo "✅ Finished $SENSOR"
        else
            echo "⚠️ Error occurred while chunking $SENSOR"
        fi
    else
        echo "⚠️ Sensor input directory not found: $SENSOR_IN_DIR — skipping."
    fi
    echo "------------------------------------------"
done

echo "🎉 All PolarH10 sensors chunked!"
echo "📍 Chunks located at: $OUT_BASE_DIR"
