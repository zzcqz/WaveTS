
# WaveTS: Wavelet MLPs Are More Efficient for Time Series Forecasting

This repository is the official implementation of WaveTS. 

The overall process can be viewed in the link： [Pipeline.pdf](https://github.com/zzcqz/WaveTS/blob/main/figure/Pipeline.pdf)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training and evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
bash dwt.sh
```

## Results

Our model achieves the following performance on :


![image](https://github.com/zzcqz/WaveTS/blob/main/figure/Result.png)

## Visualization of results

![image](https://github.com/zzcqz/WaveTS/blob/main/figure/electricity.png)

## Visualization of weight
### electricity
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/electricity_720_360_lLinear_weights.png)
### traffic
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/traffic_720_360_lLinear_weights.png)
### exchange_rate
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/exchange_rate_720_360_lLinear_weights.png)
### weather
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/weather_720_360_lLinear_weights.png)
### ETTh1
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/ETTh1_720_360_lLinear_weights.png)
### ETTh2
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/ETTh2_720_360_lLinear_weights.png)
### ETTm1
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/ETTm1_720_360_lLinear_weights.png)
### ETTm2
![image](https://github.com/zzcqz/WaveTS/blob/main/Weight_visualization/ETTm2_720_360_lLinear_weights.png)
