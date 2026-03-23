import cv2
import numpy as np

def create_synthetic_gel(output_path="synthetic_gel.png"):
    # --- 1. 初始化背景 ---
    # 根据你的原图比例，创建一张高 200，宽 480 的画布
    width, height = 480, 200
    # 你的原图背景色大概是偏浅的灰色 (像素值约 190)
    bg_color = 190 
    
    # 创建基础画布 (使用 float32 方便后续的加减运算)
    image = np.full((height, width), bg_color, dtype=np.float32)
    
    # --- 2. 定义绘制单个“凝胶条带”的函数 ---
    def draw_band(base_img, x, y, band_width, band_height, intensity, blur_ksize=(25, 9)):
        """
        x, y: 条带中心坐标
        band_width: 条带的宽度 (泳道宽度)
        band_height: 条带的厚度 (决定是粗条带还是细条带)
        intensity: 颜色深度 (0-255，值越大条带越黑)
        blur_ksize: 高斯模糊的核大小，控制条带边缘的“羽化/扩散”程度
        """
        # 创建一个全黑的掩膜
        mask = np.zeros_like(base_img)
        
        # 在掩膜上画一个白色的椭圆代表条带形状
        # 注意：OpenCV 的 ellipse 参数中，轴长是半长轴和半短轴
        cv2.ellipse(mask, (x, y), (band_width // 2, band_height // 2), 0, 0, 360, intensity, -1)
        
        # 对条带进行水平方向较强、垂直方向较弱的高斯模糊，模拟蛋白质在凝胶中的自然扩散
        mask = cv2.GaussianBlur(mask, blur_ksize, 0)
        
        # 将条带的阴影从背景中减去 (让该区域变暗)
        base_img -= mask
        return base_img

    # --- 3. 按照你的示例图复刻条带 ---
    # 示例图有 4 个泳道 (Lanes)，大致分为上下两排
    lane_x = [60, 180, 300, 420] # 4个泳道的中心X坐标
    top_y = 50                   # 上排条带的Y坐标
    bottom_y = 150               # 下排条带的Y坐标
    
    # 绘制上排条带 (从左到右)
    # 泳道1: 中等粗细，中等颜色
    image = draw_band(image, lane_x[0], top_y, band_width=80, band_height=6, intensity=100)
    # 泳道2: 非常粗，非常黑
    image = draw_band(image, lane_x[1], top_y, band_width=90, band_height=14, intensity=180)
    # 泳道3: 较细，颜色适中
    image = draw_band(image, lane_x[2], top_y, band_width=80, band_height=5, intensity=120)
    # 泳道4: 粗，非常黑
    image = draw_band(image, lane_x[3], top_y, band_width=90, band_height=12, intensity=160)

    # 绘制下排条带 (从左到右)
    # 泳道1: 几乎没有 (可以不画，或者画极弱的)
    image = draw_band(image, lane_x[0], bottom_y, band_width=60, band_height=4, intensity=10)
    # 泳道2: 比较微弱，边缘扩散严重
    image = draw_band(image, lane_x[1], bottom_y, band_width=80, band_height=8, intensity=50, blur_ksize=(35, 15))
    # 泳道3: 非常粗，非常黑
    image = draw_band(image, lane_x[2], bottom_y, band_width=90, band_height=15, intensity=190)
    # 泳道4: 中等粗细，中等颜色
    image = draw_band(image, lane_x[3], bottom_y, band_width=80, band_height=6, intensity=100)

    # --- 4. 添加全局噪声 (提升真实感) ---
    # 生成均值为0，标准差为 8 的高斯噪声
    noise = np.random.normal(0, 8, image.shape)
    image = image + noise
    
    # 将像素值限制在 0-255 范围内，并转回无符号 8 位整型
    image = np.clip(image, 0, 255).astype(np.uint8)

    # 保存并显示图片
    cv2.imwrite(output_path, image)
    print(f"✅ 合成成功！已保存为: {output_path}")

if __name__ == "__main__":
    create_synthetic_gel()