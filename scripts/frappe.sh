# default
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Default:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check Emb 30
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_30 data_emb_30
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 30 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 30 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_30 data_emb_30" > /dev/null 2>&1&


# Check Emb 50
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_50 data_emb_50
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 50 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 50 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_50 data_emb_50" > /dev/null 2>&1&



# Check K
# k_1 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 1 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#K:K_1 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&



# Check K
# k_8 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#K:K_8 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check K
# k_16 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#K:K_16 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&



# Check MoEHidden 64
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_64 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 64 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#MoEHidden:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_64 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check MoEHidden 256
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_256 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 256 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#MoEHidden:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_256 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check MoEHidden 512
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_512 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 512 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#MoEHidden:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_512 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&



# Check MoE Layer 1 (2 layer)
# k_4 MoeLayer_2 hyperLayer_3 MoEhidden 128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 1 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#MoELayer_2:K_4 MoeLayer_2 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&
 

# Check MoE Layer 3 (4 layer)
# k_4 MoeLayer_4 hyperLayer_3 MoEhidden 128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 3 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#MoELayer_4:K_4 MoeLayer_4 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check MoE Layer 4 (5 layer)
# k_4 MoeLayer_4 hyperLayer_3 MoEhidden 128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:4 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 4 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#MoELayer_5:K_4 MoeLayer_5 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check MoE Layer 5 (6 layer)
# k_4 MoeLayer_4 hyperLayer_3 MoEhidden 128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:4 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 5 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random_#MoELayer_6:K_4 MoeLayer_6 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&
 

# Check Sparse Alpha = 1
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 1:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check Sparse Alpha = 1.3
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.3 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 1.3:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check Sparse Alpha = 2.0
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2.0 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 2.0:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check Sparse Alpha = 2.3
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2.3 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 2.3:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check Sparse Alpha = 2.7
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 2.7:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&



# ----------------------------- Vertical MoE -------------------------------- #
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --C 1 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "default_vertical_sams_K_1"


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --C 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "default_vertical_sams_K_2"

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --C 3 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "default_vertical_sams_K_3"

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --C 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "default_vertical_sams_K_4"


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "baseline_dnn" 


# -------------------------------------------------
# sparseMax vertical MoE

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_moe_K_2


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_moe_K_4

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_moe_K_8


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_moe_K_16


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 32 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_moe_K_32



~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_moe_K_8_alpha_2


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset frappe --nfeat 5500 --nfield 10  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_moe_K_16_alpha_2