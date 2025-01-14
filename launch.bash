#!/bin/bash

for i in {1..5}
do
  python3 main.py > "tb_data_final_results/logs_start2_$i.log" 2>&1
done