for model_name in dwt
do
for seq_len in 720 336 192 96
do

pred_len=96
e_layers=1
  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --isGLU False
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --isGLU True
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \


  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --isGLU False
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --isGLU True
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --isGLU False
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --isGLU False
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \

    python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --isGLU True
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --isGLU False
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 0.0005 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --use_multi_scale False \
  --small_kernel_merged False \


  done
  done 
