python3  fully_supervised_models_lc.py --pretrained True --mixed_precision True --multi_lr False --learning_rate 1e-3 --scheduler True --batch_size 6 --nw 4 --gpu 3 --n_epochs 170 --model fcn --num_heads 1 --depth 1 --model_name deeplab --landcover True --rotate False --pi_rotate False --p_rotate 0.25 --scale False --size_crop 512 --eval_angle False --benchmark True --split True --save_dir /users/a/araujofj/data/save_model/ --save_best True