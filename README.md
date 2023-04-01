# BPETST

### This is an offical implementation of BPETST.

## Key Designs

:star2: **Byte Pair Encoding**: segmentation of time series are encoded into new tokens which are served as input tokens to Transformer.

## 项目目录
```
BPETST
│   README.md  
│
└───data_provider
│   │   data_factory.py
│   │   data_loader.py
│  
└───dataset
│   │   ETTh1.csv
│   │   ETTh2.csv
│   │   ...
│   
└───exp
│    │   exp_basic.py
│  
└───layers
│       │   BPETST_backbone.py
│       │   Discretization.py
│       │   Embed.py
│  
└───utils
│       │   masking.py
│       │   metrics.py
│       │   timefeatures.py
│       │   tools.py

```
TODO:\
- :white_check_mark 
- :white_large_square 

## Dataset 数据集介绍
###电力变压器数据集 (ETDataset)
![ETT](./figures/ett.png "ETDataSet Target图示")
该数据集提供了两年的数据，每个数据点每分钟记录一次（用 m 标记），它们分别来自中国同一个省的两个不同地区，分别名为ETTm1和ETTm2。每个数据集包含2年 365天 24小时 60分钟 = 1,051,200数据点。 此外，还提供一个小时级别粒度的数据集变体使用（用 h* 标记），即ETTh1和ETTh2。 每个数据点均包含8维特征，包括数据点的记录日期、预测值“油温”以及6个不同类型的外部负载值。
![ETT](./figures/ett_mean.png "ETDataSet 各列意义")
###Time Series Forecasting on ETT Benchmarks
![ETT](./figures/benchmarks.png "ETDataSet 各列意义")

## Getting Started
