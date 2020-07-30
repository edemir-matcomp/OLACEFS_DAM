export CUDA_VISIBLE_DEVICES=0

dataset_path='2_Classification_DAM/data/sentinel/2019'
output_path='2_Classification_DAM/results/'${dataset_path##*/}

#channels='rgb'
#dataset_path='/home/users/edemir/barragem_detection/data/DATASET_BARRAGEM_LASTONE2/new_sentinel/processed/2019'
#dataset_path='/home/users/edemir/barragem_detection/data/DATASET_BARRAGEM_LASTONE2/new_sentinel/processed_add_ndwi/2019'
#dataset_path='/home/users/edemir/barragem_detection/data/DATASET_BARRAGEM_LASTONE2/new_sentinel/processed_only_ndwi/2019'

num_epochs=100 
channels='all'
finetune=0
feature_extract=0
path_classes=2
optimizer='adam'
 
#mkdir logs_runs
for batch_size in 16
do
    for method in 'resnet' 'alexnet' 'vgg' 'squeezenet' 'inception' 'densenet'
    #for method in 'densenet'
    do
        for learning_rate in 0.01 0.001 0.0001
        #for learning_rate in 0.0001
        do
        
            # Create a directory for logging
            mkdir -p $output_path'/'$method\_$learning_rate\_$optimizer\_$channels
                
            script -c "python3 code/main.py \
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --num_epochs $num_epochs \
            --channels $channels \
            --method $method \
            --finetune $finetune \
            --learning_rate $learning_rate \
            --dataset_path $dataset_path \
            --path_classes $path_classes \
            --path_tb $output_path'/'$method\_$learning_rate\_$optimizer\_$channels \
            " $output_path/$method\_$learning_rate\_$optimizer\_$channels/logs_runs_$method\_$learning_rate\_$optimizer
            #logs_runs/log_$(date +%Y-%m-%d_%H.%M)_$learning_rate\_$method\_$weight_decay\_$momentum\_$finetune
        done
    done  
done

# --path_tb logs_runs/log_$(date +%Y-%m-%d_%H.%M)_$learning_rate\_$method\_$weight_decay\_$momentum\_$finetune \
