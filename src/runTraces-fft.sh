#!/bin/bash

gpu=Tesla

declare -a apps=( simpleCUFFT )

declare -A execApps
execApps["simpleCUFFT"]="./1dfft " 
 

 for app in "${apps[@]}"; do    
    nvcc -ccbin gcc-4.8 -lcufft ./simpleCUFFT/1dfft.cu -o ./simpleCUFFT/1dfft 

    if [[ "${app}" == "simpleCUFFT" ]]; then
        for i in `seq 8192 1024 131126`; do
			{ time nvprof --print-gpu-trace --csv -u s ./simpleCUFFT/${execApps["${app}"]} ${i} 0 2> ./simpleCUFFT/temp ; } 2> ./simpleCUFFT/tempTime
            
            cat ./simpleCUFFT/temp | awk -v var=$i '{print var"," $0}' | grep $gpu >> ./traces/${app}-traces.csv
            cat ./simpleCUFFT/tempTime | xargs -n6 | sed -e 's\real\\g' -e 's\user\,\g' -e 's\sys\,\g' -e 's\m\:\g' -e 's\s\\g' | awk -v var=$i '{print var"," $0}' >> ./traces/${app}-time.csv
        done
    fi
done



