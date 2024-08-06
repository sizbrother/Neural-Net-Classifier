#!/bin/bash

mkdir -p experiment_suite/$@

echo "Beginning experiments for current GP setup..."
declare -a data=("generated" "circle" "banana" "sphere")
for selected_data in "${data[@]}";
do
mkdir -p experiment_suite/$@/$selected_data
for n in {1..5};
do
    curr_test="Executing $selected_data Test $n..."
    echo $curr_test
    python3 project4.py $selected_data.csv > experiment_suite/$@/$selected_data/run_$n.txt
    strings experiment_suite/$@/$selected_data/run_$n.txt | grep "^The Average Accuracy = " | cut -d " " -f5 >> experiment_suite/$@/$selected_data/averages.txt
done
done

echo "Completed Testing Suite"
