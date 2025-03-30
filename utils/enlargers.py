import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def enlarge_image_nearest_neighbor(image, scale_factor=2):

    h, w = image.shape
    enlarged = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_NEAREST,
    )
    return Image.fromarray(enlarged)


def enlarge_image_bilinear(image, scale_factor=2):

    h, w = image.shape
    enlarged = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_LINEAR,
    )
    return Image.fromarray(enlarged)


def enlarge_image_bicubic(image, scale_factor=2):

    h, w = image.shape
    enlarged = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_CUBIC,
    )
    return Image.fromarray(enlarged)


def enlarge_image_lanczos(image, scale_factor=2):

    h, w = image.shape
    enlarged = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return Image.fromarray(enlarged)


def enlarge_image_spline(image, scale_factor=2):

    # Use scipy's spline interpolation (order=3 for cubic spline)
    enlarged = zoom(image, scale_factor, order=3)

    # Ensure output is in valid range
    enlarged = np.clip(enlarged, 0, 255).astype(np.uint8)
    return Image.fromarray(enlarged)


def enlarge_image_fft(image, scale_factor=2):

    # Convert to float
    img_float = image.astype(np.float32)

    # Compute 2D FFT
    img_fft = np.fft.fft2(img_float)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Create larger FFT with zero padding
    h, w = img_fft_shifted.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    enlarged_fft = np.zeros((new_h, new_w), dtype=complex)

    # Place the original FFT in the center
    h_offset = (new_h - h) // 2
    w_offset = (new_w - w) // 2
    enlarged_fft[h_offset : h_offset + h, w_offset : w_offset + w] = img_fft_shifted

    # Scale to maintain energy
    enlarged_fft *= scale_factor**2

    # Inverse FFT
    enlarged_fft_shifted = np.fft.ifftshift(enlarged_fft)
    enlarged_image = np.real(np.fft.ifft2(enlarged_fft_shifted))

    # Normalize and convert to uint8
    enlarged_image = np.clip(
        (enlarged_image - enlarged_image.min())
        / (enlarged_image.max() - enlarged_image.min())
        * 255,
        0,
        255,
    )

    return Image.fromarray(enlarged_image.astype(np.uint8))


def enlarge_image_gradients(image, scale_factor=2):

    # Initial bicubic upscale
    h, w = image.shape
    upscaled = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_CUBIC,
    )

    # Compute gradients of original image
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Upscale gradients
    grad_x_up = cv2.resize(
        grad_x,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_LINEAR,
    )
    grad_y_up = cv2.resize(
        grad_y,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_LINEAR,
    )

    # Scale gradients to account for resolution change
    grad_x_up *= scale_factor
    grad_y_up *= scale_factor

    # Compute gradient magnitude for edge-aware blending
    grad_mag = np.sqrt(grad_x_up**2 + grad_y_up**2)
    grad_mag = grad_mag / (np.max(grad_mag) + 1e-10)

    # Apply edge-preserving filter with gradients as guide
    result = cv2.ximgproc.guidedFilter(
        guide=upscaled.astype(np.float32),
        src=upscaled.astype(np.float32),
        radius=2,
        eps=1e-4,
    )

    # Edge enhancement based on gradient magnitude
    edge_enhanced = upscaled.astype(
        np.float32
    ) + grad_mag * 0.5 * grad_mag * upscaled.astype(np.float32)

    # Blend based on gradient magnitude
    alpha = np.clip(grad_mag * 0.8, 0, 1)
    result = (1 - alpha) * result + alpha * edge_enhanced

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def enlarge_image_nedi(image):

    # Get original dimensions
    h, w = image.shape

    # Create output image (2x scaling)
    output = np.zeros((h * 2, w * 2), dtype=np.uint8)

    # First upscale using bicubic interpolation as initialization
    temp = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Apply NEDI enhancement
    output = nedi_enhance(temp, image)

    return Image.fromarray(output)


def nedi_enhance(hr_image, lr_image, window_size=4):

    # Edge detection on low-resolution image
    edges = cv2.Canny(lr_image, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    # Upscale edges map
    edges_hr = cv2.resize(
        edges, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # Apply directional filtering on edge regions
    result = hr_image.copy()

    # Create edge-aware enhancement using guided filter
    if edges_hr.max() > 0:  # Only if edges are detected
        # Apply edge-preserving guided filter
        result = cv2.ximgproc.guidedFilter(
            guide=hr_image, src=hr_image, radius=window_size, eps=1e-6
        )

    return result
