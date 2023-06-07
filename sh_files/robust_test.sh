arr=('0.15' '0.2')
for std in ${arr[@]};
do
python train_ssl.py --eval_only --noise ${std} --eval_path 'checkpoint_best_svm.pth.tar' --comment 'ours_robust' --seed 0
done