from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 初始化 Flask 應用
app = Flask(__name__)

# 主頁，顯示參數輸入表單
@app.route('/')
def index():
    return render_template('index.html')

# 處理表單提交，進行線性回歸並顯示結果
@app.route('/result', methods=['POST'])
def result():
    try:
        # 獲取用戶輸入參數
        a = float(request.form['a'])
        b = float(request.form['b'])
        num_points = int(request.form['num_points'])
        noise = float(request.form['noise'])
        
        # 生成數據集
        np.random.seed(0)
        X = np.random.rand(num_points, 1) * 10  # 隨機生成 X 數據
        y = a * X + b + np.random.randn(num_points, 1) * noise  # 添加噪聲的 Y 數據
        
        # 使用線性回歸模型
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # 計算回歸評分 (R² 分數)
        r2 = r2_score(y, y_pred)
        
        # 繪製數據點和回歸線
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Data Points')
        ax.plot(X, y_pred, color='red', label='Regression Line')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'Linear Regression: a={a}, b={b}, noise={noise}')
        ax.legend()

        # 將圖表保存為 base64 字符串
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        # 渲染結果頁面，顯示回歸結果
        return render_template('index.html', a=a, b=b, num_points=num_points, noise=noise,
                               slope=model.coef_[0][0], intercept=model.intercept_[0], r2=r2,
                               plot_url=plot_url)
    except Exception as e:
        return f"Error: {str(e)}"

# 啟動應用
if __name__ == '__main__':
    app.run(debug=True)
