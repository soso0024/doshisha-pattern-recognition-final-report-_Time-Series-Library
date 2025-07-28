# Problem
Please implement a time series classification method on Heartbeat dataset　(https://www.timeseriesclassification.com/description.php?Dataset=Heartbeat) using TimesNet and Crossformer implemented in the time series library (https://github.com/thuml/Time-Series-Library). So, you will have two results, one by TimesNet and the other by Crossformer.

## Commands
- Training command for TimesNet
```bash
uv run python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --train_epochs 30 \
  --gpu_type mps \
  --num_workers 0
  ```

- Training command for Crossformer
  - ~change the model parameter to `--model Crossformer`~

```bash
uv run python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model Crossformer \
  --data UEA \
  --e_layers 2 \
  --d_model 256 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --train_epochs 30 \
  --gpu_type mps \
  --num_workers 0
```

- Test command for TimesNet
  - change the training parameter to `--is_training 0`

```bash
uv run python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --gpu_type mps \
  --num_workers 0
```

- Test command for Crossformer
```bash
uv run python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model Crossformer \
  --data UEA \
  --e_layers 2 \
  --d_model 256 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 1 \
  --gpu_type mps \
  --num_workers 0
```


# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**
 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

Note: 

(1) About classification: Since we include all five tasks in a unified code base, the accuracy of each subtask may fluctuate but the average performance can be reproduced (even a bit better). We have provided the reproduced checkpoints [here](https://github.com/thuml/Time-Series-Library/issues/494).

(2) About anomaly detection: Some discussion about the adjustment strategy in anomaly detection can be found [here](https://github.com/thuml/Anomaly-Transformer/issues/14). The key point is that the adjustment strategy corresponds to an event-level metric.
