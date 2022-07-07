import numpy as np
from tqdm import tqdm
from PIL import Image

SIGMA_SPACE_SQUARE = 10
SIGMA_COLOR_SQUARE = 32

weight_cache = {} # 計算結果のcache
def weight(i, j, m, n, f):
    key = "{},{},{},{}".format(i, j, m, n)
    if key in weight_cache:
        return weight_cache[key]

    space_weight = np.exp(-(((i - m) ** 2 + (j - n) ** 2) / (2 * SIGMA_SPACE_SQUARE**2)))
    color_weight = np.exp(-((np.linalg.norm(f[i, j] - f[m, n]) ** 2) / (2 * SIGMA_COLOR_SQUARE**2)))

    weight_cache[key] = space_weight * color_weight
    return space_weight * color_weight

BILATERAL_RANGE = 3 * SIGMA_SPACE_SQUARE

def bilateral_pixel(i, j, max_i, max_j, f):
    min_m = i - (BILATERAL_RANGE // 2) if i - (BILATERAL_RANGE // 2) >= 0 else 0
    max_m = i + (BILATERAL_RANGE // 2) if i + (BILATERAL_RANGE // 2) <= max_i else max_i
    min_n = j - (BILATERAL_RANGE // 2) if j - (BILATERAL_RANGE // 2) >= 0 else 0
    max_n = j + (BILATERAL_RANGE // 2) if j + (BILATERAL_RANGE // 2) <= max_j else max_j

    ans = np.array([0,0,0], dtype=np.float64)
    acc_w = 0
    
    for m in range(min_m, max_m):
        for n in range(min_n, max_n):
            ans += f[m, n] * weight(i, j, m, n, f)
            acc_w += weight(i, j, m, n, f)
    
    return ans / acc_w

def bilateral(src, result_src):
    f = np.array(Image.open(src), dtype=np.float64)

    # cacheのリセット
    global weight_cache
    weight_cache = {}

    ans = []
    for i in tqdm(range(len(f))):
        acc = []
        for j in range(len(f[0])):
            acc.append(bilateral_pixel(i, j, len(f), len(f[0]), f))
        ans.append(acc)

    new_img = Image.fromarray(np.array(ans, dtype=np.uint8))
    new_img.save(result_src)

def detail(src, smoothed, result_src):
    f_src = np.array(Image.open(src), dtype=np.float64)
    f_smoothed = np.array(Image.open(smoothed), dtype=np.float64)

    ans = []
    for i in tqdm(range(len(f_src))):
        acc = []
        for j in range(len(f_src[0])):
            r = f_src[i,j][0] - f_smoothed[i,j][0]
            g = f_src[i,j][1] - f_smoothed[i,j][1]
            b = f_src[i,j][2] - f_smoothed[i,j][2]
            acc.append(np.array([r if 0 <= r <= 255 else 0, g if 0 <= g <= 255 else 0, b if 0 <= b <= 255 else 0], dtype=np.float64))
        ans.append(acc)

    new_img = Image.fromarray(np.array(ans, dtype=np.uint8))
    new_img.save(result_src)

def enhanced(src, detail, result_src):
    f_src = np.array(Image.open(src), dtype=np.float64)
    f_smoothed = np.array(Image.open(detail), dtype=np.float64)

    ans = []
    for i in tqdm(range(len(f_src))):
        acc = []
        for j in range(len(f_src[0])):
            r = f_src[i,j][0] + 3 * f_smoothed[i,j][0]
            g = f_src[i,j][1] + 3 * f_smoothed[i,j][1]
            b = f_src[i,j][2] + 3 * f_smoothed[i,j][2]
            acc.append(np.array([r if 0 <= r <= 255 else 0 if r < 0 else 255, g if 0 <= g <= 255 else 0 if g < 0 else 255, b if 0 <= b <= 255 else 0 if b < 0 else 255], dtype=np.float64))
        ans.append(acc)

    new_img = Image.fromarray(np.array(ans, dtype=np.uint8))
    new_img.save(result_src)

def main():
    bilateral("src.jpg", "smoothed.jpg")
    detail("src.jpg", "smoothed.jpg", "detail.jpg")
    enhanced("src.jpg", "detail.jpg", "enhanced.jpg")

if __name__ == "__main__":
    main()
