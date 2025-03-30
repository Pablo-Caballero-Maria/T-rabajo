import bm3d
import cv2
import numpy as np


def denoise_image_median(image):
    # Apply median filter for denoising
    # The kernel size controls the amount of smoothing
    denoised = cv2.medianBlur(image, 11)

    return denoised


def denoise_image_lee(image, window_size=7, damping_factor=1.0):

    img_float = image.astype(np.float32)
    h, w = img_float.shape
    padded_img = np.pad(img_float, window_size // 2, mode="reflect")
    result = np.zeros_like(img_float)

    for i in range(h):
        for j in range(w):
            # Extract local window
            window = padded_img[i : i + window_size, j : j + window_size]

            # Calculate statistics
            mean = np.mean(window)
            variance = np.var(window)

            # Estimate noise variance (assumes noise is proportional to signal)
            noise_var = (mean * mean) * 0.25  # Typical assumption for SAR

            # Calculate weight (larger variance = less filtering)
            weight = variance / (variance + noise_var / damping_factor)

            # Apply filter
            center_pixel = img_float[i, j]
            result[i, j] = mean + weight * (center_pixel - mean)

    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_image_frost(image, window_size=7, damping_factor=2.0):

    img_float = image.astype(np.float32)
    h, w = img_float.shape
    padded_img = np.pad(img_float, window_size // 2, mode="reflect")
    result = np.zeros_like(img_float)

    # Create distance array for kernel weighting
    y, x = np.ogrid[
        -(window_size // 2) : (window_size // 2) + 1,
        -(window_size // 2) : (window_size // 2) + 1,
    ]
    distance = np.sqrt(x * x + y * y)

    for i in range(h):
        for j in range(w):
            # Extract window
            window = padded_img[i : i + window_size, j : j + window_size]

            # Calculate local statistics
            mean = np.mean(window)
            if mean == 0:  # Avoid division by zero
                mean = 1e-10

            # Coefficient of variation (normalized measure of dispersion)
            coef_var = np.var(window) / (mean * mean)

            # Create frost kernel
            kernel = np.exp(-damping_factor * coef_var * distance)
            kernel = kernel / np.sum(kernel)  # Normalize

            # Apply filter
            result[i, j] = np.sum(window * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_image_kuan(image, window_size=7, damping_factor=1.0):

    img_float = image.astype(np.float32)
    h, w = img_float.shape
    padded_img = np.pad(img_float, window_size // 2, mode="reflect")
    result = np.zeros_like(img_float)

    # Estimate global noise level (could be improved)
    global_mean = np.mean(img_float)
    noise_var = 0.25 * global_mean * global_mean  # Typical for SAR

    for i in range(h):
        for j in range(w):
            # Extract window
            window = padded_img[i : i + window_size, j : j + window_size]

            # Calculate local statistics
            mean = np.mean(window)
            if mean == 0:
                mean = 1e-10

            variance = np.var(window)

            # Calculate the signal-to-noise ratio
            cu = max(0, (variance - noise_var) / (noise_var + mean * mean))

            # Weighting factor
            weight = cu / (1 + cu)

            # Apply filter
            center_pixel = img_float[i, j]
            result[i, j] = mean + weight * (center_pixel - mean)

    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_image_wiener(image, window_size=7):

    # Convert to float and avoid log(0)
    img_float = image.astype(np.float32) + 1e-10

    # Log transform to convert multiplicative noise to additive
    log_img = np.log(img_float)

    # Pad image
    padded_img = np.pad(log_img, window_size // 2, mode="reflect")
    h, w = log_img.shape
    result = np.zeros_like(log_img)

    # Estimate global noise variance in log domain
    noise_var = np.var(log_img) * 0.1  # Approximation

    for i in range(h):
        for j in range(w):
            # Extract window
            window = padded_img[i : i + window_size, j : j + window_size]

            # Local mean and variance
            local_mean = np.mean(window)
            local_var = np.var(window)

            # Apply Wiener filter
            if local_var > noise_var:
                result[i, j] = local_mean + (1 - noise_var / local_var) * (
                    log_img[i, j] - local_mean
                )
            else:
                result[i, j] = local_mean

    # Convert back from log domain
    result = np.exp(result)

    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_image_non_local_means(image, search_window=21, patch_size=7, h=10.0):

    # Convert to float
    img_float = image.astype(np.float32) + 1e-10

    # Apply log transform to convert multiplicative noise to additive
    log_img = np.log(img_float)

    # OpenCV implementation is faster than a direct implementation
    # For SAR images, we need to adjust the h parameter (filtering strength)
    # We use a higher h value for log-transformed SAR images

    # Estimate noise from homogeneous region
    noise_sigma = np.std(log_img) * 0.5  # Scale factor for log-domain SAR

    # Adjust h parameter based on noise level
    h_adjusted = noise_sigma * h

    # Apply non-local means in log domain
    # Using fastNlMeansDenoising which is optimized for grayscale images
    # h parameter needs to be squared for OpenCV implementation
    denoised_log = cv2.fastNlMeansDenoising(
        src=np.uint8(
            (log_img - np.min(log_img)) * 255 / (np.max(log_img) - np.min(log_img))
        ),
        h=h_adjusted**2,
        templateWindowSize=patch_size,
        searchWindowSize=search_window,
    )

    # Rescale back to original log range
    denoised_log = denoised_log.astype(np.float32) / 255 * (
        np.max(log_img) - np.min(log_img)
    ) + np.min(log_img)

    # Transform back from log domain
    result = np.exp(denoised_log)

    return np.clip(result, 0, 255).astype(np.uint8)


def estimate_sar_noise(image, patch_size=16):

    # Find most homogeneous patch to estimate noise
    h, w = image.shape
    min_var = float("inf")

    # Sample patches to find homogeneous region
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = image[i : i + patch_size, j : j + patch_size]
            var = np.var(patch)
            if var < min_var:
                min_var = var

    # For SAR images with multiplicative noise,
    # standard deviation is approximately 0.2-0.3 in homogeneous areas
    return max(np.sqrt(min_var), 0.05)


def sar_bm3d(image, noise_std):

    # Apply log transform to convert multiplicative noise to additive
    log_image = np.log(image + 1e-10)

    # Apply standard BM3D with adjusted parameters for SAR
    denoised_log = bm3d.bm3d(log_image, noise_std)

    # Convert back from log domain
    denoised = np.exp(denoised_log) - 1e-10
    denoised = np.clip(denoised, 0, 1)

    return denoised


def denoise_image_bm3d(image):

    # Convert to float in range [0, 1] for processing
    image_float = image.astype(np.float32) / 255.0

    # Estimate noise standard deviation for SAR image
    # For SAR images, we can estimate noise based on homogeneous regions
    noise_std = estimate_sar_noise(image_float)

    # Apply BM3D with SAR adaptations
    denoised_float = sar_bm3d(image_float, noise_std)

    # Convert back to uint8
    denoised = np.clip(denoised_float * 255, 0, 255).astype(np.uint8)

    return denoised
