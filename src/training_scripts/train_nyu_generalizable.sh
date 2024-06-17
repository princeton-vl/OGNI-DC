# model configs
GRU_iters=5
optim_layer_input_clamp=1.0
depth_activation_format='exp'

# training configs
batch_size=12
lr=0.001
grad_loss_weight=1.0
intermediate_loss_weight=1.0
mask_out_rate=0.5

port=$(shuf -i 8001-40000 -n 1)

python main.py --dir_data ../datasets/nyudepthv2_h5 --data_name NYU --split_json ../data_json/nyu.json \
    --gpus 0 --tcp_port $port  \
    --lr $lr --batch_size $batch_size --milestones 18 24 28 32 36 --epochs 36 \
    --loss 1.0*SeqL1+1.0*SeqL2+$grad_loss_weight*SeqGradL1 --intermediate_loss_weight $intermediate_loss_weight --training_depth_mask_out_rate $mask_out_rate \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --log_dir ../experiments/ \
    --save "train_nyu_generalizable"