GRU_iters=5
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'

ckpt=../checkpoints/NYU_best_performance.pt

python main.py --dir_data ../datasets/nyudepthv2_h5 --data_name NYU --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --test_only --test_augment $test_augment --pretrain $ckpt \
    --log_dir ../experiments/ \
    --save "test_nyu" \
    --save_result_only


