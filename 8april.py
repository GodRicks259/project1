import numpy as np
import matplotlib.pyplot as plt

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  
    kh, kw = kernel.shape
    h, w = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Используем reflect для лучшего качества на границах
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel_flipped)
    
    return output

def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
   
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / np.sum(kernel)

def gradient_angle(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
 
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    return angle

def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:

    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            ang = angle[i, j]
            mag = magnitude[i, j]
            
            # Определяем соседей в зависимости от направления
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                # Горизонтальное направление (0°)
                neighbors = (magnitude[i, j-1], magnitude[i, j+1])
            elif 22.5 <= ang < 67.5:
                # Диагональное (45°)
                neighbors = (magnitude[i-1, j+1], magnitude[i+1, j-1])
            elif 67.5 <= ang < 112.5:
                # Вертикальное (90°)
                neighbors = (magnitude[i-1, j], magnitude[i+1, j])
            else:  # 112.5 <= ang < 157.5
                # Анти-диагональное (135°)
                neighbors = (magnitude[i-1, j-1], magnitude[i+1, j+1])
            
            if mag >= neighbors[0] and mag >= neighbors[1]:
                suppressed[i, j] = mag
    
    return suppressed

def double_threshold(suppressed: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15) -> np.ndarray:

    # Автоматическое определение порогов на основе гистограммы
    max_val = suppressed.max()
    high_thresh = max_val * high_ratio
    low_thresh = high_thresh * low_ratio
    
    h, w = suppressed.shape
    strong = 255
    weak = 50
    
    result = np.zeros_like(suppressed, dtype=np.uint8)
    
    strong_i, strong_j = np.where(suppressed >= high_thresh)
    weak_i, weak_j = np.where((suppressed >= low_thresh) & (suppressed < high_thresh))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    changed = True
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if result[i, j] == weak:
                    # Проверяем 8 соседей
                    neighbors = result[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == strong):
                        result[i, j] = strong
                        changed = True
    
    result[result != strong] = 0
    result[result == strong] = 255
    
    return result

def canny_edge_detector(image: np.ndarray, gauss_size: int = 5, sigma: float = 1.4,
                        low_ratio: float = 0.05, high_ratio: float = 0.15) -> dict:
    """
    Полный детектор границ Кэнни
    """
   
    gauss = gaussian_kernel(gauss_size, sigma)
    blurred = convolve2d(image, gauss)
    
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    gx = convolve2d(blurred, sobel_x)
    gy = convolve2d(blurred, sobel_y)
    
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = gradient_angle(gx, gy)
    
    
    suppressed = non_maximum_suppression(magnitude, angle)
    
    
    edges = double_threshold(suppressed, low_ratio, high_ratio)
    
    return {
        'blurred': blurred,
        'gx': gx,
        'gy': gy,
        'magnitude': magnitude,
        'angle': angle,
        'suppressed': suppressed,
        'edges': edges
    }

img = np.zeros((120, 180), dtype=np.float64)
img[20:90, 30:90] = 180       
img[40:100, 110:160] = 255    
img[95:105, 20:160] = 120    


result = canny_edge_detector(img, gauss_size=5, sigma=1.4, low_ratio=0.05, high_ratio=0.15)


def normalize(x: np.ndarray) -> np.ndarray:
    if x.max() == x.min():
        return x
    return (x - x.min()) / (x.max() - x.min())


plt.figure(figsize=(15, 10))


plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Оригинал')
plt.axis('off')


plt.subplot(2, 4, 2)
plt.imshow(result['blurred'], cmap='gray')
plt.title('Гауссово размытие')
plt.axis('off')


plt.subplot(2, 4, 3)
plt.imshow(normalize(np.abs(result['gx'])), cmap='gray')
plt.title('Собель X (вертикальные)')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(normalize(np.abs(result['gy'])), cmap='gray')
plt.title('Собель Y (горизонтальные)')
plt.axis('off')


plt.subplot(2, 4, 5)
plt.imshow(normalize(result['magnitude']), cmap='gray')
plt.title('Мощность градиента')
plt.axis('off')


plt.subplot(2, 4, 6)
plt.imshow(result['angle'], cmap='hsv', vmin=0, vmax=180)
plt.title('Направление градиента')
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')


plt.subplot(2, 4, 7)
plt.imshow(normalize(result['suppressed']), cmap='gray')
plt.title('Подавление немаксимумов')
plt.axis('off')


plt.subplot(2, 4, 8)
plt.imshow(result['edges'], cmap='gray')
plt.title('Детектор Кэнни (финал)')
plt.axis('off')

plt.tight_layout()
plt.show()


print(f"Статистика:")
print(f"  Мощность градиента: min={result['magnitude'].min():.2f}, max={result['magnitude'].max():.2f}")
print(f"  Углы: 0-180 градусов")
print(f"  Граничных пикселей: {np.sum(result['edges'] > 0)} из {img.size} ({100*np.sum(result['edges']>0)/img.size:.1f}%)")