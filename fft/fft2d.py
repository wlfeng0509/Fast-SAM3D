import cv2
import numpy as np
from scipy.stats import linregress  
import os

def preprocess_for_fft_masked(image_path, target_size=256):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"{image_path}")

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.imread(image_path, 0)
        if gray is None: raise ValueError("Image read error")
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(gray.shape[1] - x, w + 2 * pad)
        h = min(gray.shape[0] - y, h + 2 * pad)
        crop_gray = gray[y:y + h, x:x + w]
        crop_alpha = alpha[y:y + h, x:x + w]
    else:
        crop_gray = gray
        crop_alpha = np.ones_like(gray) * 255


    resized_gray = cv2.resize(crop_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    resized_alpha = cv2.resize(crop_alpha, (target_size, target_size), interpolation=cv2.INTER_AREA)


    mask = (resized_alpha > 0).astype(np.float32)
    object_pixels = resized_gray[resized_alpha > 0]

    if object_pixels.size > 0:
        mean_val = np.mean(object_pixels)
    else:
        mean_val = 128.0

    img_filled = np.ones_like(resized_gray, dtype=np.float32) * mean_val
    np.copyto(img_filled, resized_gray.astype(np.float32), where=(resized_alpha > 0))
    row_win = np.hanning(target_size)
    col_win = np.hanning(target_size)
    window2d = np.outer(row_win, col_win)

    processed_img = img_filled * window2d

    return processed_img



def calculate_hfer_robust(image_path, radius_ratio=0.15):
    try:
        img = preprocess_for_fft_masked(image_path)
    except Exception as e:
        print(f"Error (HFER): {e}")
        return 0.0

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude[crow, ccol] = 0 

    total_energy = np.sum(magnitude) + 1e-8
    r = int(min(rows, cols) * radius_ratio)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol) ** 2 + (y - crow) ** 2
    mask_low = mask_area <= r * r

    high_freq_energy = np.sum(magnitude[~mask_low])
    return high_freq_energy / total_energy



def calculate_slope_robust(image_path):

    try:
        img = preprocess_for_fft_masked(image_path)
    except Exception as e:
        print(f"Error (Slope): {e}")
        return 0.0

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2


    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    magnitude[crow, ccol] = 0

    y, x = np.ogrid[:rows, :cols]
    r_grid = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2).astype(int)
    max_radius = int(min(rows, cols) // 2)

    radial_sum = np.bincount(r_grid.ravel(), weights=magnitude.ravel())
    pixel_count = np.bincount(r_grid.ravel())

    radial_sum = radial_sum[1:max_radius]
    pixel_count = pixel_count[1:max_radius]
    pixel_count[pixel_count == 0] = 1
    radial_profile = radial_sum / pixel_count

    freqs = np.arange(1, max_radius)
    valid_mask = radial_profile > 1e-10

    log_f = np.log(freqs[valid_mask])
    log_amp = np.log(radial_profile[valid_mask])

    if len(log_f) < 5: 
        return 0.0

    slope, intercept, r_value, p_value, std_err = linregress(log_f, log_amp)
    return abs(slope)




def normalize_to_uint8(img_data):
    min_val = np.min(img_data)
    max_val = np.max(img_data)
    if max_val - min_val < 1e-8:
        return np.zeros(img_data.shape, dtype=np.uint8)

    norm_img = (img_data - min_val) / (max_val - min_val)
    return (norm_img * 255).astype(np.uint8)


def crop_transparent_area(img, margin=20):
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        coords = cv2.findNonZero(a)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            h_img, w_img = img.shape[:2]
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w_img, x + w + margin)
            y2 = min(h_img, y + h + margin)

            crop_img = img[y1:y2, x1:x2]
            crop_a = a[y1:y2, x1:x2]

            crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2GRAY)
            crop_gray[crop_a == 0] = 0

            return crop_gray
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def save_frequency_analysis(image_path, filter_radius=30, margin=20, output_dir=None):
    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), f"{name_no_ext}_analysis")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img_raw is None:
        print(f"Error:  {image_path}")
        return

    img = crop_transparent_area(img_raw, margin=margin)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    cv2.imwrite(os.path.join(output_dir, "0_Original_Cropped.png"), img)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    spectrum_uint8 = normalize_to_uint8(magnitude_spectrum)
    spectrum_color = cv2.applyColorMap(spectrum_uint8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(output_dir, "1_Frequency_Spectrum.png"), spectrum_color)

    mask_low = np.zeros((rows, cols), np.uint8)
    mask_low[crow - filter_radius:crow + filter_radius, ccol - filter_radius:ccol + filter_radius] = 1

    mask_high = np.ones((rows, cols), np.uint8)
    mask_high[crow - filter_radius:crow + filter_radius, ccol - filter_radius:ccol + filter_radius] = 0

    fshift_low = fshift * mask_low
    img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_low)))
    img_low_uint8 = normalize_to_uint8(img_low)
    cv2.imwrite(os.path.join(output_dir, "2_Spatial_LowFreq_Structure.png"), img_low_uint8)

    fshift_high = fshift * mask_high
    img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_high)))
    img_high_uint8 = normalize_to_uint8(img_high)
    cv2.imwrite(os.path.join(output_dir, "3_Spatial_HighFreq_Gray.png"), img_high_uint8)

    img_high_heatmap = cv2.applyColorMap(img_high_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "4_Spatial_HighFreq_Heatmap.png"), img_high_heatmap)

    img_low_bgr = cv2.cvtColor(img_low_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_low_bgr, 0.6, img_high_heatmap, 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, "5_Analysis_Overlay.png"), overlay)

