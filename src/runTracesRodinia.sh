#!/bin/bash

#declare -a apps=(  myocyte particlefilter srad streamcluster bfs b+tree cfd dwt2d huffman hybridsort kmeans)

declare -a apps=( backprop  gaussian heartwall hotspot hotspot3D  lud lavaMD myocyte mummergpu nn nw pathfinder )

declare -A execApps
execApps["backprop"]="./backprop " 
execApps["gaussian"]="./gaussian " 
execApps["heartwall"]="./heartwall " 
execApps["hotspot"]="./hotspot " 
execApps["hotspot3D"]="./3D " 
execApps["lud"]="cuda/lud_cuda " 
execApps["myocyte"]="./myocyte.out " 
execApps["nw"]="./needle " 
execApps["pathfinder"]="./pathfinder " 

cd rodinia/cuda/

for app in "${apps[@]}"; do 
    mkdir -p ../../logs/${app}
    cd ${app}
    #make clean; make
    if [[ "${app}" == "backprop" ]]; then
        for i in `seq 8192 8192 65536`; do
            nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["backprop"]} $i 2> ../../../logs/${app}/${app}-$i.csv
        done
    fi
    
    if [[ "${app}" == "gaussian" ]]; then
        for i in 16 32 64 128 `seq 256 256 2048 `; do
            nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["gaussian"]} -f ../../data/gaussian/matrix$i.txt 2> ../../../logs/${app}/${app}-f-$i.csv
        done
        for i in 16 32 64 128 `seq 256 256 2048 ` ; do
            nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["gaussian"]} -s $i 2> ../../../logs/${app}/${app}-s-$i.csv
        done
    fi
    
    if [[ "${app}" == "hotspot" ]]; then
        for i in 64 512 1024; do
            for j in 2 4 8 16 32 64 128 256 512 `seq 1024 1024 8192 `; do
                nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["hotspot"]} $i 2 $j ../../data/hotspot/temp_$i ../../data/hotspot/power_$i output.out 2> ../../../logs/${app}/${app}-$i-$j.csv
            done
        done
    fi
    
    if [[ "${app}" == "hotspot3D" ]]; then        
        for i in 2 4 8; do
            for j in 2 4 8 16 32 64 128 256 512 `seq 1024 1024 8192 `; do
                nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["hotspot3D"]} 512 8 $j ../../data/hotspot3D/power_512x$i ../../data/hotspot3D/temp_512x$i output.out 2> ../../../logs/${app}/${app}-$i-$j.csv
            done
        done
    fi      
    
    if [[ "${app}" == "lavaMD" ]]; then
        for i in `seq 10 5 100`; do
            nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["lavaMD"]} -boxes1d $i 2> ../../../logs/${app}/${app}-$i.csv
        done
    fi
    
    if [[ "${app}" == "lud" ]]; then
        for i in `seq 1024 1024 8192`; do
            nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["lud"]} -s $i -v 2> ../../../logs/${app}/${app}-$i.csv
        done
    fi
    
    if [[ "${app}" == "needle" ]]; then        
        for i in `seq 1024 1024 8192`; do
            for j in `seq 10 10 100 `; do
                nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["needle"]} $i $j 0 2> ../../../logs/${app}/${app}-$i-$j.csv
            done
        done
    fi 
    
    if [[ "${app}" == "pathfinder" ]]; then 
        for i in `seq 10000000 10000000 100000000`; do
            for j in `seq 10 10 100`; do
                for k in 2 4 8 16 32 64; do
                    nvprof  --metrics all --events all --print-gpu-trace --csv -u s ${execApps["pathfinder"]} $i $j $k 0 2> ../../../logs/${app}/${app}-$i-$j-$k.csv
                done
            done
        done
    fi     
  
    cd ..
done
    



