GRU_iters=5
test_augment=1
optim_layer_input_clamp=100.0
depth_activation_format='linear'
depth_downsample_method='min'
pred_confidence_input=1

ckpt=../checkpoints/KITTI_generalization.pt

python main.py --dir_data ../datasets/kitti/kitti_depth --data_name KITTIDC --split_json ../data_json/kitti_dc_test.json \
    --patch_height 240 --patch_width 1216 --lidar_lines 64 \
    --gpus 0 --max_depth 90.0 \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --depth_downsample_method $depth_downsample_method --pred_confidence_input $pred_confidence_input \
    --test_only --test_augment $test_augment --pretrain $ckpt \
    --log_dir ../experiments/ \
    --save "test_kitti" \
    --save_result_only