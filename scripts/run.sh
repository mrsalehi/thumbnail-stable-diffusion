
# echo myuser=`whoami`
# echo COUNT_NODE=$COUNT_NODE
# echo HOSTNAMES = $HOSTNAMES
# echo hostname = `hostname`
# echo MASTER_ADDR= $MASTER_ADDR
# echo MASTER_PORT= $MASTER_PORT
hostname=`hostname`

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
# echo THEID=$THEID


eval "$(home/miniconda3/bin/conda shell.bash hook)" # init conda
conda activate sd_webui

# use slurm env variables to find the number of gpus on this node
N_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "hostname: ${hostname} master_addr: ${MASTER_ADDR} master_port: ${MASTER_PORT} node_rank: ${THEID} nnodes: ${COUNT_NODE} n_gpus: ${N_GPUS}"

echo python3 version = `python3 --version`

if [ "$DEBUG" = True ] ; then
    DATA_DIR_IMG=../data/thumbnail-imgs
    DATA_DIR_TEXT=../data/thumbnail-titles
else
    DATA_DIR_TEXT=/tmp/thumbnail-titles
    DATA_DIR_IMG=/tmp/thumbnail-imgs
    rsync -avzPh ../data/thumbnail-imgs /tmp/
    rsync -avzPh ../data/thumbnail-titles /tmp/
fi

torchrun --nproc_per_node $N_GPUS \
    --nnodes $COUNT_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $THEID \
    thumbnail_finetune.py --pretrained_model_name_or_path /home/user/stable-diffusion-webui/diffusers_ckpt \
        --train_data_dir_image $DATA_DIR_IMG \
        --train_data_dir_text $DATA_DIR_TEXT \
        --train_batch_size 3 \
        --mixed_precision no \
        --lr_scheduler constant_with_warmup \
        --lr_warmup_steps 5000 \
        --learning_rate 1e-4 \
        --output_dir /home/user/stable-diffusion-webui/runs \
        --gradient_accumulation_steps 2 \
        --logging_dir 20_kids_channels_ftuned \
        --max_train_steps 50000 \
        --save_every_steps 5000

#--mixed_precision fp16
    # thumbnail_finetune.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \