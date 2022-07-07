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

    space_weight = np.exp(-(((i - m) ** 2 + (j - n) ** 2) / (2 * SIGMA_SPACE_SQUARE)))
    color_weight = np.exp(-((np.linalg.norm(f[i, j] - f[m, n]) ** 2) / (2 * SIGMA_COLOR_SQUARE)))

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
    f = np.array(Image.open(src), dtype=np.float64)
    g = np.array(Image.open(smoothed), dtype=np.float64)

    ans = []
    for i in tqdm(range(len(f))):
        acc = []
        for j in range(len(f[0])):
            acc.append(np.array(f[i, j] - g[i, j], dtype=np.float64))
        ans.append(acc)

    new_img = Image.fromarray(np.array(ans, dtype=np.uint8))
    new_img.save(result_src)

def enhanced(src, detail, result_src):
    f = np.array(Image.open(src), dtype=np.float64)
    g = np.array(Image.open(detail), dtype=np.float64)

    ans = []
    for i in tqdm(range(len(f))):
        acc = []
        for j in range(len(f[0])):
            acc.append(np.array(f[i, j] + 3 * g[i, j], dtype=np.float64))
        ans.append(acc)

    new_img = Image.fromarray(np.array(ans, dtype=np.uint8))
    new_img.save(result_src)

def main():
    # bilateral("src.jpg", "smoothed.jpg")
    detail("src.jpg", "smoothed.jpg", "detail.jpg")
    enhanced("src.jpg", "detail.jpg", "enhanced.jpg")

if __name__ == "__main__":
    main()
