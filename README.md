
# WaveTS: Wavelet MLPs Are More Efficient for Time Series Forecasting

This repository is the official implementation of WaveTS. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training and evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
bash scripts/dwt.sh
```

## Results

Our model achieves the following performance on :


![image](https://github.com/zzcqz/WaveTS/blob/main/figure/Result.png)

## Visualization of results

![image](https://github.com/zzcqz/WaveTS/blob/main/figure/electricity.png)
![image](https://github.com/zzcqz/WaveTS/blob/main/Result_visualization/Forecast_results.png)

## Visualization of weight
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/lLinear_weights.png)

