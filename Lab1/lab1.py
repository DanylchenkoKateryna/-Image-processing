import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

IMAGE_PATH = r".\cat.jpg"

# Робота з цифровими зображеннями

def section1_basic_operations(image_path):
    print("=" * 60)
    print("СЕКЦІЯ 1: Основні операції з зображенням")
    print("=" * 60)

    # Відкриття зображення (np.fromfile для шляхів з кирилицею)
    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Зображення не знайдено: {image_path}")

    height, width, channels = img.shape
    print(f"Розмір: {width}x{height} пікселів")
    print(f"Кількість каналів: {channels}")
    print(f"Тип даних: {img.dtype}")
    print(f"Розмір масиву: {img.shape}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Секція 1: Основні операції з зображенням", fontsize=14)

    # Оригінал
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"Оригінал ({width}x{height})")
    axes[0, 0].axis('off')

    #Розділення на канали
    b, g, r = cv2.split(img)

    r_img = np.zeros_like(img)
    r_img[:, :, 2] = r
    g_img = np.zeros_like(img)
    g_img[:, :, 1] = g
    b_img = np.zeros_like(img)
    b_img[:, :, 0] = b

    axes[0, 1].imshow(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Червоний канал (R)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Зелений канал (G)")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title("Синій канал (B)")
    axes[0, 3].axis('off')

    # Гістограми каналів
    colors = ('r', 'g', 'b')
    channel_data = [r, g, b]
    labels = ['R', 'G', 'B']
    axes[1, 0].set_title("Гістограми каналів")
    for ch_data, color, label in zip(channel_data, colors, labels):
        hist = cv2.calcHist([ch_data], [0], None, [256], [0, 256])
        axes[1, 0].plot(hist, color=color, label=label)
    axes[1, 0].legend()
    axes[1, 0].set_xlim([0, 256])

    # ROI — виділення ділянки інтересу (морда кота)
    h, w = img.shape[:2]
    roi_x, roi_y = w // 4, h // 4
    roi_w, roi_h = w // 2, h // 2
    roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    axes[1, 1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"ROI ({roi_w}x{roi_h})")
    axes[1, 1].axis('off')

    # Малювання прямокутника
    img_rect = img_rgb.copy()
    cv2.rectangle(img_rect, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 3)
    axes[1, 2].imshow(img_rect)
    axes[1, 2].set_title("Прямокутник ROI")
    axes[1, 2].axis('off')

    # Поворот зображення на 15 градусів
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    axes[1, 3].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title("Поворот на 15°")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig("section1_results.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("Секція 1 завершена. Збережено: section1_results.png\n")

    return img

# Застосування фільтрів

def section2_filters(img):
    print("=" * 60)
    print("СЕКЦІЯ 2: Фільтри розмиття та різкості")
    print("=" * 60)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Розмиття ---
    # Усереднювальне розмиття
    blur_avg = cv2.blur(img, (15, 15))

    # Гаусівське розмиття
    blur_gauss = cv2.GaussianBlur(img, (15, 15), 0)

    # Медіанне розмиття
    blur_median = cv2.medianBlur(img, 15)

    # Білатеральне розмиття (зберігає краї)
    blur_bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Секція 2.1: Фільтри розмиття", fontsize=14)

    titles_blur = [
        ("Оригінал", img_rgb),
        ("Усереднювальне (15x15)", cv2.cvtColor(blur_avg, cv2.COLOR_BGR2RGB)),
        ("Гаусівське (15x15)", cv2.cvtColor(blur_gauss, cv2.COLOR_BGR2RGB)),
        ("Медіанне (15)", cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB)),
        ("Білатеральне (d=9)", cv2.cvtColor(blur_bilateral, cv2.COLOR_BGR2RGB)),
    ]

    for i, (title, image) in enumerate(titles_blur):
        row, col = divmod(i, 4)
        axes[row, col].imshow(image)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    # Порівняння деталей (кроп)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 3
    crop_size = 150
    for idx, (title, image) in enumerate(titles_blur[1:4], start=5):
        row, col = divmod(idx, 4)
        crop = image[cy:cy+crop_size, cx:cx+crop_size]
        axes[row, col].imshow(crop)
        axes[row, col].set_title(f"Кроп: {title.split()[0]}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("section2_blur.png", dpi=120, bbox_inches='tight')
    plt.show()

    # --- Підвищення різкості ---
    # Unsharp Masking
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    usm = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    # Лапласіан
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    laplacian_sharp = cv2.addWeighted(img, 1.0, cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR), -0.7, 0)

    # Bilateral + USM
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    gaussian_b = cv2.GaussianBlur(bilateral, (9, 9), 10.0)
    edge_preserve_sharp = cv2.addWeighted(bilateral, 1.5, gaussian_b, -0.5, 0)

    # Detail enhancement
    detail_enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
    fig2.suptitle("Секція 2.2: Підвищення різкості", fontsize=14)

    titles_sharp = [
        ("Оригінал", img_rgb),
        ("USM (alpha=1.5)", cv2.cvtColor(usm, cv2.COLOR_BGR2RGB)),
        ("Лапласіан", cv2.cvtColor(laplacian_sharp, cv2.COLOR_BGR2RGB)),
        ("Bilateral + USM", cv2.cvtColor(edge_preserve_sharp, cv2.COLOR_BGR2RGB)),
        ("Detail Enhance", cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2RGB)),
    ]

    for i, (title, image) in enumerate(titles_sharp):
        row, col = divmod(i, 3)
        axes2[row, col].imshow(image)
        axes2[row, col].set_title(title)
        axes2[row, col].axis('off')

    axes2[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("section2_sharpening.png", dpi=120, bbox_inches='tight')
    plt.show()

    print("Секція 2 завершена. Збережено: section2_blur.png, section2_sharpening.png\n")
    return img, blur_gauss

# Виявлення країв

def section3_edge_detection(img):
    print("=" * 60)
    print("СЕКЦІЯ 3: Виявлення країв")
    print("=" * 60)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Порогова бінаризація
    _, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Детектор Кенні
    edges_canny_low = cv2.Canny(gray, 30, 100)
    edges_canny_mid = cv2.Canny(gray, 80, 150)
    edges_canny_high = cv2.Canny(gray, 150, 250)

    # Комбінований підхід (поріг + Кенні)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_combined = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_combined = cv2.Canny(blurred, 50, 150)
    combined_result = cv2.bitwise_and(thresh_combined, edges_combined)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Секція 3: Виявлення країв", fontsize=14)

    plots = [
        ("Оригінал (сірий)", gray, 'gray'),
        ("Поріг (Binary, t=127)", thresh_binary, 'gray'),
        ("Поріг (Otsu)", thresh_otsu, 'gray'),
        ("Адаптивний поріг", thresh_adaptive, 'gray'),
        ("Кенні (30/100) - м'який", edges_canny_low, 'gray'),
        ("Кенні (80/150) - середній", edges_canny_mid, 'gray'),
        ("Кенні (150/250) - жорсткий", edges_canny_high, 'gray'),
        ("Отсу + Гаус", thresh_combined, 'gray'),
        ("Комбінований (Отсу & Кенні)", combined_result, 'gray'),
    ]

    for i, (title, image, cmap) in enumerate(plots):
        row, col = divmod(i, 3)
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("section3_edges.png", dpi=120, bbox_inches='tight')
    plt.show()

    print("Секція 3 завершена. Збережено: section3_edges.png\n")
    return img

# Згортка через NumPy

def apply_convolution_numpy(image_gray, kernel):
    """Реалізація 2D-згортки через NumPy (без padding)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image_gray, ((ph, ph), (pw, pw)), mode='reflect').astype(np.float64)
    h, w = image_gray.shape
    output = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
    return output


def section4_numpy_convolution(img):
    print("=" * 60)
    print("СЕКЦІЯ 4: Згортка через NumPy")
    print("=" * 60)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Ядра фільтрів
    kernel_blur = np.ones((5, 5), dtype=np.float64) / 25.0

    kernel_sharpen = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)

    kernel_sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    kernel_sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    print("Виконується NumPy-згортка (може зайняти кілька секунд)...")

    # Розмиття
    blurred_np = apply_convolution_numpy(gray, kernel_blur)
    blurred_np = np.clip(blurred_np, 0, 255).astype(np.uint8)

    # Різкість
    sharpened_np = apply_convolution_numpy(gray, kernel_sharpen)
    sharpened_np = np.clip(sharpened_np, 0, 255).astype(np.uint8)

    # Собель X та Y
    sobel_x = apply_convolution_numpy(gray, kernel_sobel_x)
    sobel_y = apply_convolution_numpy(gray, kernel_sobel_y)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = (sobel_mag / sobel_mag.max() * 255).astype(np.uint8)

    # Порівняння NumPy vs OpenCV
    blurred_cv2 = cv2.filter2D(gray.astype(np.uint8), -1, kernel_blur)
    sharpened_cv2 = cv2.filter2D(gray.astype(np.uint8), -1, kernel_sharpen)
    sobel_x_cv2 = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_cv2 = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
    sobel_cv2 = np.sqrt(sobel_x_cv2 ** 2 + sobel_y_cv2 ** 2)
    sobel_cv2 = (sobel_cv2 / sobel_cv2.max() * 255).astype(np.uint8)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle("Секція 4: Згортка NumPy vs OpenCV", fontsize=14)

    rows_data = [
        ("Оригінал (сірий)", gray.astype(np.uint8),
         "Розмиття (NumPy)", blurred_np,
         "Розмиття (OpenCV)", blurred_cv2,
         "Ядро розмиття", kernel_blur),
        ("Оригінал (сірий)", gray.astype(np.uint8),
         "Різкість (NumPy)", sharpened_np,
         "Різкість (OpenCV)", sharpened_cv2,
         "Ядро різкості", kernel_sharpen),
        ("Оригінал (сірий)", gray.astype(np.uint8),
         "Собель (NumPy)", sobel_mag,
         "Собель (OpenCV)", sobel_cv2,
         "Ядро Собеля X", kernel_sobel_x),
    ]

    for row_idx, (t1, i1, t2, i2, t3, i3, tk, kernel_img) in enumerate(rows_data):
        axes[row_idx, 0].imshow(i1, cmap='gray')
        axes[row_idx, 0].set_title(t1)
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(i2, cmap='gray')
        axes[row_idx, 1].set_title(t2)
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(i3, cmap='gray')
        axes[row_idx, 2].set_title(t3)
        axes[row_idx, 2].axis('off')

        im = axes[row_idx, 3].imshow(kernel_img, cmap='RdBu_r', aspect='auto')
        axes[row_idx, 3].set_title(tk)
        for ii in range(kernel_img.shape[0]):
            for jj in range(kernel_img.shape[1]):
                axes[row_idx, 3].text(jj, ii, f"{kernel_img[ii, jj]:.2f}",
                                      ha='center', va='center', fontsize=7,
                                      color='black')
        plt.colorbar(im, ax=axes[row_idx, 3])

    plt.tight_layout()
    plt.savefig("section4_numpy_convolution.png", dpi=120, bbox_inches='tight')
    plt.show()

    # Порівняння MAE
    mae_blur = np.mean(np.abs(blurred_np.astype(float) - blurred_cv2.astype(float)))
    mae_sharp = np.mean(np.abs(sharpened_np.astype(float) - sharpened_cv2.astype(float)))
    mae_sobel = np.mean(np.abs(sobel_mag.astype(float) - sobel_cv2.astype(float)))
    print(f"MAE (розмиття NumPy vs OpenCV):  {mae_blur:.4f}")
    print(f"MAE (різкість NumPy vs OpenCV):  {mae_sharp:.4f}")
    print(f"MAE (Собель NumPy vs OpenCV):    {mae_sobel:.4f}")
    print("Секція 4 завершена. Збережено: section4_numpy_convolution.png\n")


if __name__ == "__main__":
    print("Лабораторна робота №1 — Основи цифрової обробки зображень")
    print(f"Зображення: {IMAGE_PATH}\n")

    img = section1_basic_operations(IMAGE_PATH)
    img, blurred = section2_filters(img)
    section3_edge_detection(img)
    section4_numpy_convolution(img)

    print("=" * 60)
    print("Всі секції виконано успішно!")
    print("Збережені файли:")
    print("  - section1_results.png")
    print("  - section2_blur.png")
    print("  - section2_sharpening.png")
    print("  - section3_edges.png")
    print("  - section4_numpy_convolution.png")
