# SAW（序列整体）

这是论文《序列整体：一种基于长程文本查询的统一视频动作定位框架》的官方实现。[[论文链接](https://ieeexplore.ieee.org/document/10043827)]

![网络架构图](./docs/net.png)

## 项目概述

我们提出了一个统一的框架，以序列方式处理整个视频，实现端到端的长程密集视觉-语言交互。具体来说，我们设计了一个基于相关性过滤的轻量级Transformer（Ref-Transformer），它由基于相关性过滤的注意力机制和时间扩展的MLP组成。通过相关性过滤，可以高效地突出视频中与文本相关的空间区域和时间片段，然后通过时间扩展的MLP在整个视频序列中传播。这个统一框架可以应用于各种视频-文本动作定位任务，如指代视频分割、时序句子定位和时空视频定位。

## 环境要求

* Python 3.9
* PyTorch 2.1.2
* torchtext 0.18.0

## 指代视频分割

进入 `referring_segmentation` 目录以进行指代视频分割任务。

### 1. 数据集准备

从 [这里](https://kgavrilyuk.github.io/publication/actor_action/) 下载A2D Sentences数据集和J-HMDB Sentences数据集，并将视频转换为RGB帧。

对于A2D Sentences数据集，运行 `python pre_proc\video2imgs.py` 将视频转换为RGB帧。预期的目录结构如下：

```python
-a2d_sentences
    -Rename_Images
    -a2d_annotation_with_instances
    -videoset.csv
    -a2d_missed_videos.txt
    -a2d_annotation.txt
-jhmdb_sentences
    -Rename_Images
    -puppet_mask
    -jhmdb_annotation.txt
```


编辑 `json/config_$DATASET$.json` 中的 `datasets_root` 项为当前数据集路径。

运行 `python pre_proc\generate_data_list.py` 生成训练和测试数据分割。

### 2. 骨干网络

从 [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) 下载预训练的DeepLabResNet，并将其放入 `model/pretrained/` 目录。

### 3. 训练

仅使用A2D Sentences数据集进行训练，运行：


```python
python main.py --json_file=json\config_a2d_sentences.json --mode=train
```

### 4. 评估

对于A2D Sentences数据集，运行：


```python
python main.py --json_file=json\config_a2d_sentences.json --mode=test
``` 

对于JHMDB Sentences数据集，运行：

```python
python main.py --json_file=json\config_jhmdb_sentences.json --mode=test
``` 

## 时序句子定位

进入 `temporal_grounding` 目录以进行时序句子定位任务。

### 1. 数据集准备

* Charades-STA数据集：按照 [LGI4temporalgrounding](https://github.com/JonghwanMun/LGI4temporalgrounding) 下载预提取的I3D特征，按照 [2D-TAN](https://github.com/microsoft/VideoX/tree/master/2D-TAN) 下载预提取的VGG特征。

* TACoS数据集：按照 [2D-TAN](https://github.com/microsoft/VideoX/tree/master/2D-TAN) 下载预提取的C3D特征。

* ActivityNet Captions数据集：从 [这里](http://activity-net.org/challenges/2016/download.html) 下载预提取的C3D特征。

### 2. 训练和评估

配置文件位于 `./json` 目录，支持以下模型设置：

```
-config_ActivityNet_C3D_anchor.json
-config_ActivityNet_C3D_regression.json
-config_Charades-STA_I3D_anchor.json
-config_Charades-STA_I3D_regression.json
-config_Charades-STA_VGG_anchor.json
-config_Charades-STA_VGG_regression.json
-config_TACoS_C3D_anchor.json
-config_TACoS_C3D_regression.json
```

Set the `"datasets_root"` in each config file to be your feature path.

To train on different dataset with different grounding heads, run

```
python main.py --json_file=$JSON_FILE_PATH$ --mode=train
```

For evaluation, run 

```
python main.py --json_file=$JSON_FILE_PATH$ --mode=test --checkpoint=$CHECKPOINT_PATH$
```

The pretrained models and their correspondance performance are shown bellow

| Datasets     | Feature | Decoder    | Checkpoints |
|--------------|---------|------------|-------------|
| Charades-STA | I3D     | Regression        |  \[[Baidu](https://pan.baidu.com/s/1GQBkElQITd-exS1njNZrwQ) \| gj54 \]           |
| Charades-STA | I3D     | Anchor     |    \[[Baidu](https://pan.baidu.com/s/1MXZqAEBLOzauR8cOLjo3QA) \| 5j3a \]           |
| Charades-STA | VGG     | Regression |                \[[Baidu](https://pan.baidu.com/s/1Yacke_tkaAELzMY_ePyIhw) \| 52xf \]           |
| Charades-STA | VGG     | Anchor     |                \[[Baidu](https://pan.baidu.com/s/1PcIZ7QEWcYnfzne1dkMsng) \| rdmx \]          |
| ActivityNet  | C3D     | Regression |                \[[Baidu](https://pan.baidu.com/s/1zlH64seHimscTOtNry-6Ag) \| 6sbh \]            |
| ActivityNet  | C3D     | Anchor     |              \[[Baidu](https://pan.baidu.com/s/1mi8M2wBUAqskWQQqHdmi2Q) \| ysr5 \]           |
| TACOS        | C3D     | Regression |                \[[Baidu](https://pan.baidu.com/s/140m-9geYbktSRfP7Pa1rzA) \| iwx2 \]           |
| TACOS        | C3D     | Anchor     |               \[[Baidu](https://pan.baidu.com/s/1dzIIb4dKQY9t-oAF-N2sLw) \| 1ube \]           |


## Spatiotemporal Video Grounding

run `cd spatiotemporal_grounding` for spatiotemporal video grounding task. The code for spatiotemporal grounding is built on the [TubeDETR codebase](https://github.com/antoyang/TubeDETR).

### 1. Dataset

We prepare the `HC-STVG` and `VidSTG` datasets following the [TubeDETR](https://github.com/antoyang/TubeDETR). The annotation formation of the VidSTG dataset has been optimized to reduce the training memory usage. 

**videos**

VidSTG dataset: Download VidOR videos from [the VidOR dataset providers](https://xdshang.github.io/docs/vidor.html)

HC-STVG dataset: Download HC-STVG videos from [the HC-STVG dataset providers](https://github.com/tzhhhh123/HC-STVG).

Edit the item `vidstg_vid_path` in `spatiotemporal_grounding/config/vidstg.json` and the `hcstvg_vid_path` in `spatiotemporal_grounding/config/hcstvg.json` to be the current video path.

**annotations**

Download the preprocessed annotation files from \[[https://pan.baidu.com/s/1oiV9PmtRqRxxdxMvqrJj_w](https://pan.baidu.com/s/1oiV9PmtRqRxxdxMvqrJj_w), password: n6y4\]. Then put the downloaded `annotations` into `spatiotemporal_grounding`.

### 2. Training and Evaluation

To train on HC-STVG dataset, run

```
python main.py --combine_datasets=hcstvg --combine_datasets_val=hcstvg --dataset_config config/hcstvg.json --output-dir=hcstvg_result
```

To train on VidSTG dataset, run

```
python main.py --combine_datasets=vidstg --combine_datasets_val=vidstg --dataset_config config/vidstg.json --output-dir=vidstg_result
```

To evaluate on HC-STVG dataset, run:

```
python main.py --combine_datasets=hcstvg --combine_datasets_val=hcstvg --dataset_config config/hcstvg.json --output-dir=hcstvg_result --eval --resume=$CHECKPOINT_PATH$
```

To evaluate on VidSTG dataset, run

```
python main.py --combine_datasets=vidstg --combine_datasets_val=vidstg --dataset_config config/vidstg.json --output-dir=vidstg_result --eval --resume=$CHECKPOINT_PATH$
```

## Citation

```
@article{2023saw,
    title     = {Sequence as A Whole: A Unified Framework for Video Action Localization with Long-range Text Query},
    author    = {Yuting Su, Weikang Wang, Jing Liu, Shuang Ma, Xiaokang Yang},
    booktitle = {IEEE Transactions on Image Processing},
    year      = {2023}
}
```
