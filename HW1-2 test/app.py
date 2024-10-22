from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 初始化 Flask 應用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# 創建 static 文件夾如果不存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 首頁路由，顯示文件上傳表單
@app.route('/')
def index():
    return render_template('index.html')

# 處理文件上傳和預測
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # 讀取 CSV 文件
        df = pd.read_csv(file)
        
        # 處理 Date 和 y 欄位
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['y'] = df['y'].replace(',', '', regex=True).astype(float)
        df.rename(columns={'Date': 'ds'}, inplace=True)
        
        # 使用 Prophet 模型
        model = Prophet(changepoint_prior_scale=0.5)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df)
        
        # 預測未來 60 天
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)
        
        # 繪製圖表
        plt.figure(figsize=(10, 6))
        # 實際數據（黑色線條）
        plt.plot(df['ds'], df['y'], color='black', label='Actual Data')
        
        # 預測數據（藍色線條）
        plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Predicted Data')
        
        # 不確定性區間（淺藍色陰影）
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.4)
        
        # 歷史平均值（灰色虛線）
        historical_mean = df['y'].mean()
        plt.axhline(y=historical_mean, color='gray', linestyle='--', label='Historical Average')
        
        # 預測初始化點（紅色虛線）
        prediction_start = df['ds'].max()
        plt.axvline(x=prediction_start, color='red', linestyle='--', label='Forecast Initialization')
        
        # 標註 "upward trend" 的箭頭
        latest_prediction = forecast.iloc[-1]
        plt.annotate('Upward Trend', xy=(latest_prediction['ds'], latest_prediction['yhat']),
                     xytext=(latest_prediction['ds'], latest_prediction['yhat'] + 10),
                     arrowprops=dict(facecolor='green', shrink=0.05), color='green', fontsize=12)
        
        # 添加圖例、標題和標籤
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()

        # 保存圖片到 static 文件夾
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'forecast.png')
        plt.savefig(image_path)

        # 確保圖片已保存
        if os.path.exists(image_path):
            print(f"Image saved at {image_path}")

        return redirect(url_for('show_plot'))

# 顯示生成的圖表
@app.route('/plot')
def show_plot():
    image_url = url_for('static', filename='images/forecast.png')
    return render_template('plot.html', image_url=image_url)

# 啟動應用
if __name__ == '__main__':
    app.run(debug=True)
