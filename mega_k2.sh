export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH=/mnt/lustre/sjtu/home/gry10/icefall:$PYTHONPATH
export TORCH_DISTRIBUTED_DEBUG=DETAIL

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/icefall/bin/python egs/librispeech/ASR/mega_ctc/train.py \
--exp-dir egs/librispeech/ASR/mega_ctc/exp_500_att0.8 \
--lang-dir /data/icefall_librispeech/data/lang_bpe_500 \
--att-rate 0.8 \
--full-libri 1 \
--max-duration 200 \
--concatenate-cuts 0 \
--world-size 4 \
--bucketing-sampler 1 \
--start-epoch 0 \
--num-epochs 90 \
--manifest-dir /data/icefall_librispeech/data/fbank \
--embedding_dim 512 \
--hidden_dim 512 \
--ffn_hidden_dim 512 \

#cd egs/librispeech/ASR/mega_ctc
# cd egs/librispeech/ASR/conformer_ctc
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# ./conformer_ctc/train.py \
#   --exp-dir conformer_ctc/exp_500_att0.8 \
#   --lang-dir data/lang_bpe_500 \
#   --att-rate 0.8 \
#   --full-libri 1 \
#   --max-duration 200 \
#   --concatenate-cuts 0 \
#   --world-size 4 \
#   --bucketing-sampler 1 \
#   --start-epoch 0 \
#   --num-epochs 90
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="7"
#export TORCH_DISTRIBUTED_DEBUG=INFO

# python egs/librispeech/ASR/conformer_ctc/train.py \
#   --exp-dir egs/librispeech/ASR/conformer_ctc/exp_500_att0.8 \
#   --lang-dir /data/icefall_librispeech/data/lang_bpe_500 \
#   --att-rate 0.8 \
#   --full-libri 1 \
#   --max-duration 200 \
#   --concatenate-cuts 0 \
#   --world-size 4 \
#   --bucketing-sampler 1 \
#   --start-epoch 0 \
#   --num-epochs 90 \
#   --manifest-dir /data/icefall_librispeech/data/fbank \
# 这个是没问题的

