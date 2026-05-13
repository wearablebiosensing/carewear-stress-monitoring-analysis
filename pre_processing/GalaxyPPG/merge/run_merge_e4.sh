#!/bin/bash

# --- CONFIGURATION ---
# Default paths (Update these or pass them as arguments)
DATA_DIR=${1:-"/Volumes/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/Dataset"}
OUT_DIR=${2:-"/Volumes/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/4_merged_lables/EmpaticaE4"}

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Usage: ./run_merge_e4.sh [DATA_DIR] [OUT_DIR]"
    exit 1
fi

echo "🚀 Starting Empatica E4 Merging Pipeline..."
echo "📂 Input: $DATA_DIR"
echo "📂 Output: $OUT_DIR"
echo "------------------------------------------"

# List of sensors to process
SENSORS=("ACC" "HR" "BVP" "TEMP" "IBI")

for SENSOR in "${SENSORS[@]}"; do
    echo "📝 Processing Sensor: $SENSOR"
    python3 merge_galaxyppg_e4.py --sensor "$SENSOR" --data "$DATA_DIR" --out "$OUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Finished $SENSOR"
    else
        echo "⚠️ Error occurred while processing $SENSOR"
    fi
    echo "------------------------------------------"
done

echo "🎉 All Empatica E4 sensors processed!"
