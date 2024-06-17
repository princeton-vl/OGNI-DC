# model configs
GRU_iters=5
depth_activation_format='linear'
optim_layer_input_clamp=100.0
pred_confidence_input=1
depth_downsample_method='min'

# training configs
batch_size=5
lr=0.001
grad_loss_weight=1.0
intermediate_loss_weight=1.0
mask_out_rate=0.0

python main.py --dir_data ../datasets/kitti/kitti_depth --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
    --patch_height 240 --patch_width 1216 --lidar_lines 64 \
    --max_depth 90.0 --top_crop 100 --test_crop \
    --gpus 0,1,2,3,4,5,6,7 --multiprocessing \
    --lr $lr --batch_size $batch_size --milestones 50 60 70 80 90 --epochs 100 \
    --loss 1.0*SeqL1+1.0*SeqL2+$grad_loss_weight*SeqGradL1 --intermediate_loss_weight $intermediate_loss_weight --training_depth_mask_out_rate $mask_out_rate \
    --depth_downsample_method $depth_downsample_method --pred_confidence_input $pred_confidence_input \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --log_dir ../experiments/ \
    --save "train_kitti_best_performance_l1l2" --save_full