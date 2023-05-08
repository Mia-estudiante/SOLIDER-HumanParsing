#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 train.py --arch swin --imagenet-pretrain ./pretrain_model/kg_pretrain/swin_base_mask_cluster.pth --batch-size 6  --learning-rate 3e-2 --weight-decay 0 --optimizer sgd --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 150 --schp-start 120 --input-size 572,384 --log-dir ./logs/sota/base_lr3e-2
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 train.py --arch swin --imagenet-pretrain ./pretrain_model/kg_pretrain/dino.pth --batch-size 8  --learning-rate 7e-3 --weight-decay 0 --optimizer sgd --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 120 --schp-start 90 --input-size 572,384 --log-dir ./logs/sgd_phase2/swin_dino_16_7e-3_wd0_500-2_e30-e60-e30_572x384_fix1rd
# tiny
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 train.py --arch swin --imagenet-pretrain /mnt_det/xianzhe.xxz/projects/Self-Correction-Human-Parsing/networks/backbone/models_cond/dino_mask_cluster_algnmulcond.pth --batch-size 8  --learning-rate 7e-3 --weight-decay 0 --optimizer sgd --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 150 --schp-start 120 --input-size 572,384 --log-dir ./logs/aline_swin-t_ablation_sw2e-1
# base
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 train.py --arch swin --imagenet-pretrain /mnt_det/xianzhe.xxz/projects/Self-Correction-Human-Parsing/networks/backbone/models_cond/swin_base_mask_cluster_algnmulcond.pth --batch-size 6  --learning-rate 2e-3 --weight-decay 0 --optimizer sgd --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 10 --epochs 150 --schp-start 120 --input-size 572,384 --log-dir ./logs/aline_swin-b_sota_lr2e-3
# small
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 train.py --arch swin --imagenet-pretrain /mnt_det/xianzhe.xxz/projects/Self-Correction-Human-Parsing/networks/backbone/models_cond/swin_small_mask_cluster_algnmulcond.pth --batch-size 8  --learning-rate 7e-3 --weight-decay 0 --optimizer sgd --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 150 --schp-start 120 --input-size 572,384 --log-dir ./logs/aline_swin-s_sota
