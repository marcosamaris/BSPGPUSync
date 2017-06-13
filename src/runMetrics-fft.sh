#!/bin/bash

gpu=Tesla

declare -a apps=( simpleCUFFT )

declare -A execApps
execApps["simpleCUFFT"]="./1dfft " 
 

 for app in "${apps[@]}"; do    
    nvcc -ccbin gcc-4.8 -lcufft ./simpleCUFFT/1dfft.cu -o ./simpleCUFFT/1dfft 

    if [[ "${app}" == "simpleCUFFT" ]]; then
        for i in `seq 8192 1024 131126`; do
			nvprof --metrics all --events all --print-gpu-trace --csv -u s ./simpleCUFFT/${execApps["${app}"]} ${i} 0 2> ./simpleCUFFT/temp 
        
            cat ./simpleCUFFT/temp | awk -v var=$i  -v var2=$j '{print var"," var2"," $0}'  | grep $gpu >> ./metrics/${app}-metrics.csv		
        done
    fi
done



