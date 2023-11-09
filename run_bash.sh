
## maml training

python train_maml.py --datasets miniimagenet --epoch 60000  --num_shots 5 --update maml --batch_size 2 --gpu_id 5 --version 1 &
python train_maml.py --datasets miniimagenet --epoch 60000  --num_shots 5 --update maml --batch_size 2 --gpu_id 6 --version 2 &
python train_maml.py --datasets miniimagenet --epoch 60000  --num_shots 5 --update maml --batch_size 2 --gpu_id 7 --version 3 &


## test 
# python eval_meta.py --datasets CIFAR_FS --classifier euclidean --num_shots 1 --gpu_id 5 &
# python eval_meta.py --datasets CIFAR_FS --classifier euclidean --num_shots 5 --gpu_id 6 &



wait
echo "All experiments completed."