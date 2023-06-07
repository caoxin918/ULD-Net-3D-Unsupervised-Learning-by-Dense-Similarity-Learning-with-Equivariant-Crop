python train_ssl.py \
--dataset shapenet \
--data_root ./dataset/ \
--save_root results/ULD-Net/train \
--comments PretrainOnShapeNet \
--lr 0.001 \
--seed 2021 \
--batch_size_train 24 \
--batch_size_test 16 \
--batch_size_eval 16 \
--num_epoch 200 \
--augment --scale --rotateperturbation --jitter --normalize --translate --randomcrop