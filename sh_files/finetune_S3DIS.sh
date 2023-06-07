for area in $(seq 1 1 6)
do
python train_semseg.py \
        --test_area ${area} \
        --optimizer sgd \
        --scheduler cos \
        --model dgcnn_semseg \
        --log_dir dgcnn_area${area}_region \
        --restore
done
