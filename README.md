# BPETST

### This is an offical implementation of BPETST.

## Key Designs

:star2: **Byte Pair Encoding**: segmentation of time series are encoded into new tokens which are served as input tokens to Transformer.

:star2: **Channel-independence**: each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.

