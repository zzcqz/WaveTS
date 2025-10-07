export CUDA_VISIBLE_DEVICES=0

model_name=WaveTSB

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_720_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --lradj type3 \
    --dropout 0.05 \
    --batch_size 16 \
    --train_epochs 50 \
    --learning_rate 0.0003 \
    --hidden 720 \

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_720_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --lradj type3 \
    --dropout 0.05 \
    --batch_size 32 \
    --train_epochs 50 \
    --learning_rate 0.0001 \
    --hidden 360 \

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_720_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --lradj type3 \
    --dropout 0.05 \
    --batch_size 32 \
    --train_epochs 50 \
    --learning_rate 0.0001 \
    --hidden 360 \

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_720_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --lradj type3 \
    --dropout 0.05 \
    --batch_size 32 \
    --train_epochs 50 \
    --learning_rate 0.001 \
    --hidden 360 \