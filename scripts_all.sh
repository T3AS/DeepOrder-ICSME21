#!/bin/sh
start=$(date '+%d-%m-%Y %H:%M:%S')
rm -r time.csv
rm -r values.csv
rm -r output.csv

echo "$start Starting DeepOrder"
echo "$(date '+%d-%m-%Y %H:%M:%S') DeepOrder on Cisco"
./DeepOrder_on_Cisco_Dataset.py
echo "========================================"
echo "$(date '+%d-%m-%Y %H:%M:%S') DeepOrder on Iofrol"
./DeepOrder_on_iofrol_Dataset.py
echo "========================================"
echo "$(date '+%d-%m-%Y %H:%M:%S') DeepOrder on Paintcontrol"
./DeepOrder_on_paintcontrol_Dataset.py
echo "========================================"
echo "$(date '+%d-%m-%Y %H:%M:%S') DeepOrder on Gsdtsr"
./DeepOrder_on_google_Dataset.py
end=$(date '+%d-%m-%Y %H:%M:%S')
echo "$end DeepOrder Completed"
