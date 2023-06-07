python train_ssl.py \
--dataset shapenet \
--data_root ./dataset/ \
--save_root results/ULD-Net/test \
--comments test \
--batch_size_train 24 \
--batch_size_test 16 \
--batch_size_eval 16 \
--augment --scale --rotateperturbation --jitter --normalize --translate --randomcrop --eval_only
