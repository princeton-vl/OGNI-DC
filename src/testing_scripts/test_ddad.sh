GRU_iters=5
test_augment=0
optim_layer_input_clamp=100.0
depth_activation_format='linear'
depth_downsample_method='min'
pred_confidence_input=0

ckpt=../checkpoints/KITTI_generalization.pt

python main.py --dir_data ../datasets/ddad/pregenerated/val --data_name DDAD \
    --gpus 0 --max_depth 250.0 \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --depth_downsample_method $depth_downsample_method --pred_confidence_input $pred_confidence_input \
    --test_only --test_augment $test_augment --pretrain $ckpt \
    --log_dir ../experiments/ \
    --save "val_kitti_lines${lidar_lines}" \
    --save_result_only
