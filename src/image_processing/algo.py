import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def add_gaussian_noise(image, mean=0, std=25):
    """Добавляет Гауссов шум к изображению"""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def preprocess_image(image_path):
    """Выполняет полную предобработку изображения"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    
    noisy_image = add_gaussian_noise(image)
    
    _, binary_image = cv2.threshold(noisy_image, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(binary_image, None)
    
    return {
        'original': image,
        'noisy': noisy_image,
        'binary': binary_image,
        'keypoints': keypoints,
        'descriptors': descriptors
    }

def match_keypoints(data1, data2, ratio=0.75):
    """Сопоставление ключевых точек с использованием BFMatcher"""
    if data1['descriptors'] is None or data2['descriptors'] is None:
        print("⚠️ Дескрипторы не найдены!")
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(data1['descriptors'], data2['descriptors'], k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    
    return good_matches

def stitch_images(img1, img2, orientation='horizontal'):
    """Сшивает два изображения без отображения ключевых точек"""
    if orientation == 'horizontal':
        # Горизонтальная сшивка
        height = max(img1.shape[0], img2.shape[0])
        width = img1.shape[1] + img2.shape[1]
        stitched = np.zeros((height, width), dtype=np.uint8)
        stitched[:img1.shape[0], :img1.shape[1]] = img1
        stitched[:img2.shape[0], img1.shape[1]:] = img2
    else:
        # Вертикальная сшивка
        height = img1.shape[0] + img2.shape[0]
        width = max(img1.shape[1], img2.shape[1])
        stitched = np.zeros((height, width), dtype=np.uint8)
        stitched[:img1.shape[0], :img1.shape[1]] = img1
        stitched[img1.shape[0]:, :img2.shape[1]] = img2
    
    return stitched

def save_images(data1, data2, matches, output_dir='output', orientation='horizontal'):
    """Сохраняет 5 изображений в отдельные файлы"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    saved_files = []
    
    # 1. Первое изображение с шумом
    path1 = os.path.join(output_dir, f'{timestamp}_image1_noisy.png')
    cv2.imwrite(path1, data1['noisy'])
    saved_files.append(path1)
    print(f"✅ Сохранено: {path1}")
    
    # 2. Первое изображение после бинаризации
    path2 = os.path.join(output_dir, f'{timestamp}_image1_binary.png')
    cv2.imwrite(path2, data1['binary'])
    saved_files.append(path2)
    print(f"✅ Сохранено: {path2}")
    
    # 3. Второе изображение с шумом
    path3 = os.path.join(output_dir, f'{timestamp}_image2_noisy.png')
    cv2.imwrite(path3, data2['noisy'])
    saved_files.append(path3)
    print(f"✅ Сохранено: {path3}")
    
    # 4. Второе изображение после бинаризации
    path4 = os.path.join(output_dir, f'{timestamp}_image2_binary.png')
    cv2.imwrite(path4, data2['binary'])
    saved_files.append(path4)
    print(f"✅ Сохранено: {path4}")
    
    # 5. Сшивка двух бинаризированных изображений (без ключевых точек)
    stitched_image = stitch_images(data1['binary'], data2['binary'], orientation)
    path5 = os.path.join(output_dir, f'{timestamp}_stitched.png')
    cv2.imwrite(path5, stitched_image)
    saved_files.append(path5)
    print(f"✅ Сохранено: {path5}")
    
    return saved_files

def visualize_results(data1, data2, matches):
    """Визуализация результатов обработки и сопоставления"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(data1['noisy'], cmap='gray')
    axes[0, 0].set_title(f'Изображение 1 с шумом\nКлючевых точек: {len(data1["keypoints"])}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(data1['binary'], cmap='gray')
    axes[0, 1].set_title('Бинаризация (Оцу)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(data2['noisy'], cmap='gray')
    axes[1, 0].set_title(f'Изображение 2 с шумом\nКлючевых точек: {len(data2["keypoints"])}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(data2['binary'], cmap='gray')
    axes[1, 1].set_title('Бинаризация (Оцу)')
    axes[1, 1].axis('off')
    
    # Для визуализации в окне показываем сшивку
    stitched = stitch_images(data1['binary'], data2['binary'])
    axes[0, 2].imshow(stitched, cmap='gray')
    axes[0, 2].set_title(f'Сшивка изображений\nСопоставлений: {len(matches)}')
    axes[0, 2].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    img1_path = 'pictures/1.png'
    img2_path = 'pictures/4.png'
    output_dir = 'output'
    orientation = 'horizontal'  # 'horizontal' или 'vertical'
    
    print("🔄 Обработка первого изображения...")
    data1 = preprocess_image(img1_path)
    
    print("🔄 Обработка второго изображения...")
    data2 = preprocess_image(img2_path)
    
    print("🔍 Сопоставление ключевых точек...")
    matches = match_keypoints(data1, data2)
    
    print(f"✅ Найдено {len(matches)} хороших сопоставлений")
    
    print("\n💾 Сохранение 5 изображений...")
    saved_files = save_images(data1, data2, matches, output_dir, orientation)
    
    print(f"\n📊 Статистика:")
    print(f"Ключевых точек (изобр. 1): {len(data1['keypoints'])}")
    print(f"Ключевых точек (изобр. 2): {len(data2['keypoints'])}")
    print(f"Сопоставлений: {len(matches)}")
    if len(matches) > 0:
        avg_distance = np.mean([m.distance for m in matches])
        print(f"Среднее расстояние: {avg_distance:.2f}")
    
    print(f"\n📁 Все изображения сохранены в папку: {output_dir}/")
    print(f"📄 Всего файлов: {len(saved_files)}")
    
    visualize_results(data1, data2, matches)

if __name__ == "__main__":
    main()