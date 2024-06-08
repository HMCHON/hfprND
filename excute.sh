#!/bin/bash

### Train CODE
## 첫 번째 루프: augment 옵션 없이 실행
#for i in {1,3,5,7,9,11}
#do
#  for j in {17..28}
#  do
#    python train.py --n_frame=$i --model="model$j" --dataset="data/hofs_e/csves_same_stride" --save_folder_name="0529_model${j}_NA" --n_category=3
#  done
#done

# 두 번째 루프: augment 옵션과 sigma 추가하여 실행
for i in {1,3,5,7,9,11}
do
  for j in {5..28}
  do
    python train.py --n_frame=$i --model="model$j" --dataset="data/hofs_e/csves_same_stride" --save_folder_name="0606_model${j}_A" --augment=True --sigma=10 --n_category=3
  done
done

# Test CODE
for i in {5..28}
do
  case $i in
    17|18|19|20|21|22|23|24|25|26|27|28)
      python test.py --model_path="0606_model${i}_NA" --dataset="data/hofs_e/csves_same_stride" --group=True --n_category=3
      python test.py --model_path="0606_model${i}_A" --dataset="data/hofs_e/csves_same_stride" --group=True --n_category=3
      ;;
    *)
      python test.py --model_path="0606_model${i}_NA" --dataset="data/hofs_e/csves_same_stride" --group=False
      python test.py --model_path="0606_model${i}_A" --dataset="data/hofs_e/csves_same_stride" --group=False
      ;;
  esac
done



# Prediction CODE
for i in {5..28}
do
  case $i in
    17|18|19|20|21|22|23|24|25|26|27|28)
      python prediction.py --model_path="0606_model${i}_NA" --dataset="data/hofs_e/prediction/rtmpose/prediction_csves_same_stride1" --group=True --n_category=3
      python prediction.py --model_path="0606_model${i}_A" --dataset="data/hofs_e/prediction/rtmpose/prediction_csves_same_stride1" --group=True --n_category=3
      ;;
    *)
      python prediction.py --model_path="0606_model${i}_NA" --dataset="data/hofs_e/prediction/rtmpose/prediction_csves_same_stride1" --group=False
      python prediction.py --model_path="0606_model${i}_A" --dataset="data/hofs_e/prediction/rtmpose/prediction_csves_same_stride1" --group=False
      ;;
  esac
done

# Plot CODE
for i in {5..28}
do
  python plot.py --model_path="0606_model${i}_NA"
  python plot.py --model_path="0606_model${i}_A"
done