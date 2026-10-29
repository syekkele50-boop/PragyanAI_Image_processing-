import io
import math
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import ndimage

st.set_page_config(page_title="PragyanAI - Image Processing Tutorial Lab", page_icon="🖼️", layout="wide")

MODULES = {
    "Basic Operations": [
        "Grayscale", "Resize", "Crop", "Rotate", "Flip", "Image Information"
    ],
    "Intensity & Contrast": [
        "Brightness", "Contrast", "Gamma Correction", "Histogram Equalization",
        "CLAHE", "Normalize Intensity", "Invert"
    ],
    "Color Processing": [
        "RGB", "HSV", "LAB", "Grayscale", "Red Channel", "Green Channel",
        "Blue Channel", "Saturation", "Colorize Grayscale"
    ],
    "Filtering & Smoothing": [
        "Mean Blur", "Gaussian Blur", "Median Blur", "Bilateral Filter",
        "Box Filter", "Non-local Means Denoising", "Sharpen"
    ],
    "Edge Detection": [
        "Sobel", "Laplacian", "Canny", "Scharr", "Laplacian of Gaussian"
    ],
    "Morphological Operations": [
        "Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient",
        "Top Hat", "Black Hat"
    ],
    "Image Enhancement": [
        "Unsharp Mask", "CLAHE Enhancement", "Detail Enhancement",
        "Denoise + Sharpen", "White Balance (Gray World)"
    ],
    "Image Segmentation": [
        "Binary Threshold", "Adaptive Threshold", "Otsu Threshold",
        "Color Range Mask", "Watershed Demo"
    ],
    "Frequency Domain": [
        "Magnitude Spectrum", "Low-pass Filter", "High-pass Filter",
        "Band-pass Filter", "Frequency Reconstruction"
    ],
    "Feature Extraction": [
        "Harris Corners", "Shi-Tomasi Corners", "ORB Keypoints", "HOG Visualization"
    ],
    "Image Restoration": [
        "Add Gaussian Noise", "Add Salt & Pepper Noise", "Gaussian Denoising",
        "Inpainting Demo"
    ],
    "Advanced Techniques": [
        "Perspective Transform", "Affine Transform", "Pyramid Downsample",
        "Pyramid Upsample", "Image Gradient Magnitude"
    ],
    "Medical Image Examples": [
        "Medical Grayscale", "Medical CLAHE", "Medical Denoising",
        "Medical Edge Map", "Medical Thresholding"
    ],
}

EXPLANATIONS = {
    "Grayscale": "Converts color channels to a single intensity channel. Useful when color is not essential or when preparing many classical vision pipelines.",
    "Resize": "Changes spatial dimensions. Keep aspect ratio when appropriate; resizing can remove fine detail or introduce interpolation artifacts.",
    "Brightness": "Adds an intensity offset. Clipping can hide details in bright or dark regions.",
    "Contrast": "Scales distance from a midpoint. Strong contrast can reveal structure but may clip highlights and shadows.",
    "Gamma Correction": "Applies a nonlinear intensity mapping. Gamma below 1 brightens darker regions; gamma above 1 darkens them.",
    "Histogram Equalization": "Redistributes grayscale intensities to use a wider range. It can improve global contrast but may amplify noise.",
    "CLAHE": "Contrast Limited Adaptive Histogram Equalization works on local tiles and limits amplification. Often useful for uneven illumination.",
    "Gaussian Blur": "Smooths using a Gaussian-weighted neighborhood. It reduces high-frequency noise but also softens edges.",
    "Median Blur": "Replaces each pixel with a neighborhood median. It is especially useful for salt-and-pepper noise.",
    "Bilateral Filter": "Smooths within similar colors while preserving many edges; slower than a simple blur.",
    "Canny": "A multi-stage edge detector using gradients, non-maximum suppression, and hysteresis thresholds.",
    "Erosion": "Shrinks bright foreground regions. Useful for removing small bright artifacts or separating touching objects.",
    "Dilation": "Expands bright foreground regions. Useful for filling small gaps and connecting nearby components.",
    "Magnitude Spectrum": "Displays the magnitude of the Fourier transform. The center represents low frequencies after shifting.",
    "Low-pass Filter": "Keeps low spatial frequencies and suppresses high frequencies. The result is smoother and may be blurred.",
    "High-pass Filter": "Suppresses low frequencies and retains rapid changes, emphasizing edges and fine detail.",
    "Otsu Threshold": "Automatically chooses a global threshold by maximizing separation between two intensity classes.",
    "Unsharp Mask": "Adds a scaled high-frequency detail layer back to the image to increase perceived sharpness.",
}

def to_rgb_array(pil_image):
    return np.array(ImageOps.exif_transpose(pil_image).convert("RGB"))

def as_displayable(arr):
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return (arr.astype(np.uint8) * 255)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr)
            if arr.min() >= 0 and arr.max() <= 1:
                arr = arr * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def apply_operation(rgb, module, operation, p):
    src = rgb.copy()
    g = gray(src)
    note = EXPLANATIONS.get(operation, "Explore the output and compare it with the original. Parameter changes may affect detail, noise, edges, and downstream model performance.")

    if operation == "Image Information":
        return src, f"Shape: {src.shape} | dtype: {src.dtype} | min: {src.min()} | max: {src.max()} | mean: {src.mean():.2f}", note
    if operation == "Grayscale" or operation == "Medical Grayscale":
        return g, None, note
    if operation == "Resize":
        scale = p["scale"]
        h, w = src.shape[:2]
        return cv2.resize(src, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC), None, note
    if operation == "Crop":
        h, w = src.shape[:2]
        margin = p["crop_margin"]
        y1, y2 = int(h*margin), max(int(h*(1-margin)), int(h*margin)+1)
        x1, x2 = int(w*margin), max(int(w*(1-margin)), int(w*margin)+1)
        return src[y1:y2, x1:x2], None, "Crops a centered region. Cropping changes the field of view and may remove important context."
    if operation == "Rotate":
        h, w = src.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), p["angle"], 1.0)
        return cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT), None, "Rotation is a geometric transform; the output canvas remains the original size."
    if operation == "Flip":
        return cv2.flip(src, p["flip_code"]), None, "Flips the image horizontally or vertically."
    if operation == "Brightness":
        return cv2.convertScaleAbs(src, alpha=1, beta=p["brightness"]), None, note
    if operation == "Contrast":
        return cv2.convertScaleAbs(src, alpha=p["contrast"], beta=0), None, note
    if operation == "Gamma Correction":
        gamma = p["gamma"]
        lut = np.array([np.clip(((i/255.0)**gamma)*255, 0, 255) for i in range(256)], dtype=np.uint8)
        return cv2.LUT(src, lut), None, note
    if operation in ("Histogram Equalization", "CLAHE", "CLAHE Enhancement", "Medical CLAHE"):
        if operation == "Histogram Equalization":
            return cv2.equalizeHist(g), None, note
        lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=p["clip_limit"], tileGridSize=(p["tile_size"], p["tile_size"]))
        enhanced = cv2.merge((cl.apply(l), a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB), None, note
    if operation == "Normalize Intensity":
        return cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX), None, "Linearly maps observed values to the full 0–255 range."
    if operation == "Invert":
        return 255-src, None, "Maps each channel value to 255 minus the original value."
    if operation in ("RGB", "HSV", "LAB"):
        if operation == "RGB": return src, None, "Displays the original RGB representation."
        if operation == "HSV": return cv2.cvtColor(src, cv2.COLOR_RGB2HSV), None, "HSV separates hue, saturation, and value. OpenCV's HSV display values are not directly equivalent to RGB."
        return cv2.cvtColor(src, cv2.COLOR_RGB2LAB), None, "LAB separates lightness from two color-opponent channels."
    if operation in ("Red Channel", "Green Channel", "Blue Channel"):
        idx = {"Red Channel":0, "Green Channel":1, "Blue Channel":2}[operation]
        return src[:,:,idx], None, "Shows one RGB channel as a grayscale intensity map."
    if operation == "Saturation":
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        return hsv[:,:,1], None, "Shows the saturation channel from HSV."
    if operation == "Colorize Grayscale":
        return cv2.applyColorMap(g, cv2.COLORMAP_TURBO), None, "Applies a pseudocolor map for visualization; colors do not represent new measured information."
    if operation in ("Mean Blur", "Box Filter"):
        k = p["kernel"]
        return cv2.blur(src, (k,k)), None, "Averaging filters reduce local variation but soften edges."
    if operation == "Gaussian Blur":
        k = p["kernel"] | 1
        return cv2.GaussianBlur(src, (k,k), p["sigma"]), None, note
    if operation == "Median Blur":
        return cv2.medianBlur(src, p["kernel"] | 1), None, note
    if operation == "Bilateral Filter":
        return cv2.bilateralFilter(src, p["diameter"], p["sigma_color"], p["sigma_space"]), None, note
    if operation == "Non-local Means Denoising":
        return cv2.fastNlMeansDenoisingColored(src, None, p["denoise_h"], p["denoise_h"], 7, 21), None, "Non-local means averages similar patches across the image. It can preserve structure better than simple blur but costs more compute."
    if operation == "Sharpen" or operation == "Unsharp Mask":
        blur = cv2.GaussianBlur(src, (0,0), p["sigma"])
        return cv2.addWeighted(src, 1+p["amount"], blur, -p["amount"], 0), None, note
    if operation in ("Sobel", "Scharr", "Laplacian", "Laplacian of Gaussian", "Medical Edge Map"):
        if operation == "Sobel":
            sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            out = cv2.magnitude(sx, sy)
        elif operation == "Scharr":
            sx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
            sy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
            out = cv2.magnitude(sx, sy)
        else:
            base = cv2.GaussianBlur(g, (0,0), p["sigma"]) if operation == "Laplacian of Gaussian" else g
            out = np.abs(cv2.Laplacian(base, cv2.CV_32F))
        return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), None, "Gradient-based operators highlight intensity changes; noise can create strong false edges."
    if operation == "Canny":
        return cv2.Canny(g, p["threshold1"], p["threshold2"]), None, note
    if operation in ("Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient", "Top Hat", "Black Hat"):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["morph_kernel"], p["morph_kernel"]))
        if operation == "Erosion": out = cv2.erode(g, k, iterations=p["iterations"])
        elif operation == "Dilation": out = cv2.dilate(g, k, iterations=p["iterations"])
        else:
            op = {"Opening":cv2.MORPH_OPEN, "Closing":cv2.MORPH_CLOSE, "Morphological Gradient":cv2.MORPH_GRADIENT, "Top Hat":cv2.MORPH_TOPHAT, "Black Hat":cv2.MORPH_BLACKHAT}[operation]
            out = cv2.morphologyEx(g, op, k, iterations=p["iterations"])
        return out, None, "Morphology processes shapes using a structuring element. Results depend on kernel shape and size."
    if operation == "Binary Threshold" or operation == "Medical Thresholding":
        _, out = cv2.threshold(g, p["threshold"], 255, cv2.THRESH_BINARY)
        return out, None, "Global thresholding separates pixels based on one intensity cutoff."
    if operation == "Otsu Threshold":
        _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return out, None, note
    if operation == "Adaptive Threshold":
        block = p["adaptive_block"] | 1
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, p["adaptive_c"]), None, "Adaptive thresholding estimates a local threshold, useful under uneven lighting."
    if operation == "Color Range Mask":
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        lower = np.array([p["hue_min"], 40, 30], dtype=np.uint8)
        upper = np.array([p["hue_max"], 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper), None, "Creates a mask for an HSV hue interval. Hue limits may need adjustment for the target image."
    if operation == "Watershed Demo":
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.35*dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(src, cv2.COLOR_RGB2BGR), markers)
        out = src.copy()
        out[markers == -1] = [255, 0, 0]
        return out, None, "Watershed is sensitive to foreground/background markers. This demonstration uses automatic threshold-based markers."
    if operation in ("Magnitude Spectrum", "Low-pass Filter", "High-pass Filter", "Band-pass Filter", "Frequency Reconstruction"):
        f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
        mag = 20*np.log(np.abs(f)+1)
        h, w = g.shape
        cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy-cy)**2+(xx-cx)**2)
        radius = p["frequency_radius"]
        low = (dist <= radius)
        if operation == "Magnitude Spectrum":
            return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), None, "Fourier magnitude visualizes spatial frequencies; bright central areas commonly represent low-frequency content."
        if operation == "Low-pass Filter": mask = low
        elif operation == "High-pass Filter": mask = ~low
        else: mask = (dist >= radius) & (dist <= min(radius*2, max(h,w)/2))
        filtered = f * mask
        rec = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
        rec = cv2.normalize(rec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return rec, None, "Frequency filtering edits Fourier coefficients. Hard masks can cause ringing; smoother masks are often preferable in production."
    if operation == "Harris Corners":
        corners = cv2.cornerHarris(np.float32(g), 2, 3, 0.04)
        out = src.copy()
        out[corners > 0.01*corners.max()] = [255,0,0]
        return out, None, "Harris corner detection identifies locations with strong intensity changes in two directions."
    if operation == "Shi-Tomasi Corners":
        pts = cv2.goodFeaturesToTrack(g, maxCorners=200, qualityLevel=0.01, minDistance=7)
        out = src.copy()
        if pts is not None:
            for pt in np.int32(pts):
                x,y = pt.ravel()
                cv2.circle(out, (x,y), 3, (255,0,0), -1)
        return out, None, "Shi–Tomasi selects corners using the minimum eigenvalue of the local gradient matrix."
    if operation == "ORB Keypoints":
        orb = cv2.ORB_create(nfeatures=500)
        kp = orb.detect(g, None)
        out = cv2.drawKeypoints(src, kp, None, color=(255,0,0), flags=0)
        return out, None, "ORB finds keypoints and binary descriptors useful for matching and tracking."
    if operation == "HOG Visualization":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        vis = np.zeros_like(g)
        cell = 8
        for y in range(0, g.shape[0]-cell, cell):
            for x in range(0, g.shape[1]-cell, cell):
                m = mag[y:y+cell, x:x+cell]
                a = ang[y:y+cell, x:x+cell]
                angle = int(np.mean(a)//20)*20
                length = int(min(7, np.mean(m)/10))
                cv2.line(vis, (x+cell//2-length, y+cell//2), (x+cell//2+length, y+cell//2), 255, 1)
        return vis, None, "This is a simplified gradient-orientation visualization, not a full normalized HOG descriptor."
    if operation == "Add Gaussian Noise":
        rng = np.random.default_rng(p["seed"])
        noise = rng.normal(0, p["noise_sigma"], src.shape)
        return np.clip(src.astype(np.float32)+noise, 0, 255).astype(np.uint8), None, "Synthetic noise helps demonstrate denoising methods; it is not a substitute for measuring real sensor noise."
    if operation == "Add Salt & Pepper Noise":
        rng = np.random.default_rng(p["seed"])
        out = src.copy()
        mask = rng.random(src.shape[:2])
        out[mask < p["noise_amount"]/2] = 0
        out[mask > 1-p["noise_amount"]/2] = 255
        return out, None, "Salt-and-pepper noise replaces random pixels with black or white values."
    if operation == "Gaussian Denoising" or operation == "Medical Denoising":
        return cv2.fastNlMeansDenoisingColored(src, None, p["denoise_h"], p["denoise_h"], 7, 21), None, "Denoising reduces noise but can remove subtle image details."
    if operation == "Inpainting Demo":
        # Demonstration mask: small central circle; use with caution on real data.
        mask = np.zeros(g.shape, np.uint8)
        cv2.circle(mask, (g.shape[1]//2, g.shape[0]//2), max(3, min(g.shape)//18), 255, -1)
        return cv2.inpaint(src, mask, 3, cv2.INPAINT_TELEA), None, "Demonstrates filling a small masked region. Inpainting invents plausible pixels and must not be used to fabricate scientific evidence."
    if operation == "Detail Enhancement":
        return cv2.detailEnhance(src, sigma_s=10, sigma_r=0.15), None, "Detail enhancement increases local detail and can also amplify texture or noise."
    if operation == "Denoise + Sharpen":
        den = cv2.fastNlMeansDenoisingColored(src, None, p["denoise_h"], p["denoise_h"], 7, 21)
        blur = cv2.GaussianBlur(den, (0,0), 1.2)
        return cv2.addWeighted(den, 1.5, blur, -0.5, 0), None, "A simple denoise-then-sharpen pipeline; inspect for halos and lost detail."
    if operation == "White Balance (Gray World)":
        mean = src.mean(axis=(0,1))
        target = mean.mean()
        scale = target / np.maximum(mean, 1)
        return np.clip(src*scale, 0, 255).astype(np.uint8), None, "Gray-world white balance assumes average scene color should be neutral; this assumption fails for strongly colored scenes."
    if operation == "Perspective Transform":
        h,w = src.shape[:2]
        inset = p["perspective_inset"]
        src_pts = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
        dst_pts = np.float32([[inset,inset],[w-1-inset,0],[0,h-1-inset],[w-1,h-1]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(src, M, (w,h)), None, "Perspective transforms map four source points to four destination points. Real rectification requires selecting meaningful corners."
    if operation == "Affine Transform":
        h,w = src.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2), p["angle"], p["scale"])
        M[0,2] += p["translate_x"]; M[1,2] += p["translate_y"]
        return cv2.warpAffine(src, M, (w,h), borderMode=cv2.BORDER_REFLECT), None, "Affine transforms combine rotation, scaling, and translation while preserving straight lines and parallelism."
    if operation == "Pyramid Downsample":
        return cv2.pyrDown(src), None, "Pyramid downsampling smooths then reduces image dimensions."
    if operation == "Pyramid Upsample":
        return cv2.pyrUp(src), None, "Pyramid upsampling increases dimensions but cannot restore lost detail."
    if operation == "Image Gradient Magnitude":
        sx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        sy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        return cv2.normalize(cv2.magnitude(sx,sy), None, 0,255,cv2.NORM_MINMAX).astype(np.uint8), None, "Gradient magnitude estimates how quickly intensity changes spatially."
    return src, None, note

def show_histogram(image, title):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    arr = np.asarray(image)
    if arr.ndim == 2:
        ax.hist(arr.ravel(), bins=64, range=(0,255))
    else:
        for i, label in enumerate(["R", "G", "B"]):
            ax.hist(arr[:,:,i].ravel(), bins=64, range=(0,255), histtype="step", label=label)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

def image_bytes(arr, fmt="PNG"):
    arr = as_displayable(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        im = Image.fromarray(arr, "RGB")
    else:
        im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()

st.title("PragyanAI - Image Processing Tutorial Laboratory")
st.image("sample_image.jpg")
st.caption("Upload • Explore • Adjust • Compare • Understand • Download")
with st.sidebar:
    st.header("Learning Modules")
    module = st.selectbox("Choose a topic", list(MODULES.keys()))
    operation = st.selectbox("Choose a technique", MODULES[module])
    st.divider()
    st.markdown("**Suggested learning path**")
    st.markdown("1. Basic operations\n2. Intensity and contrast\n3. Color and filtering\n4. Edges and morphology\n5. Segmentation\n6. Frequency domain\n7. Features and restoration")
    st.caption("Educational demo. Validate all preprocessing choices for your own dataset and task.")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp","tif","tiff","webp"])
if uploaded is None:
    st.info("Upload an image to begin. You can use a natural image, document, satellite image, X-ray, or MRI image. For medical data, use appropriately de-identified images and preserve the original file.")
    st.markdown("### What you can learn")
    cards = st.columns(4)
    for col, title, body in zip(cards, ["Enhancement","Filtering","Frequency analysis","Computer vision"], [
        "Brightness, contrast, gamma, histogram equalization and CLAHE.",
        "Mean, Gaussian, median, bilateral filters and denoising.",
        "Fourier spectrum, low-pass, high-pass and band-pass filters.",
        "Edges, morphology, thresholding, keypoints and gradients."]):
        with col:
            st.subheader(title)
            st.write(body)
    st.stop()

try:
    pil = Image.open(uploaded)
    original = to_rgb_array(pil)
except Exception as e:
    st.error(f"Could not read this image: {e}")
    st.stop()

st.sidebar.subheader("Parameters")
p = {
    "scale": st.sidebar.slider("Resize scale", 0.10, 2.00, 0.75, 0.05),
    "crop_margin": st.sidebar.slider("Crop margin (fraction per side)", 0.00, 0.40, 0.10, 0.01),
    "angle": st.sidebar.slider("Rotation / affine angle", -180, 180, 0, 5),
    "flip_code": st.sidebar.selectbox("Flip direction", [("Horizontal",1),("Vertical",0),("Both",-1)], format_func=lambda x:x[0])[1],
    "brightness": st.sidebar.slider("Brightness offset", -100, 100, 20, 5),
    "contrast": st.sidebar.slider("Contrast multiplier", 0.10, 3.00, 1.25, 0.05),
    "gamma": st.sidebar.slider("Gamma", 0.20, 3.00, 0.80, 0.05),
    "clip_limit": st.sidebar.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5),
    "tile_size": st.sidebar.slider("CLAHE tile size", 2, 16, 8),
    "kernel": st.sidebar.select_slider("Filter kernel size", options=[1,3,5,7,9,11,15,21,31], value=5),
    "sigma": st.sidebar.slider("Gaussian sigma", 0.1, 10.0, 1.5, 0.1),
    "sigma_color": st.sidebar.slider("Bilateral sigma color", 5, 150, 75, 5),
    "sigma_space": st.sidebar.slider("Bilateral sigma space", 5, 150, 75, 5),
    "diameter": st.sidebar.slider("Bilateral diameter", 3, 15, 9, 2),
    "denoise_h": st.sidebar.slider("Denoising strength", 1, 30, 10),
    "amount": st.sidebar.slider("Sharpen amount", 0.1, 3.0, 1.0, 0.1),
    "threshold": st.sidebar.slider("Threshold", 0, 255, 127),
    "threshold1": st.sidebar.slider("Canny lower threshold", 0, 255, 80),
    "threshold2": st.sidebar.slider("Canny upper threshold", 0, 255, 160),
    "morph_kernel": st.sidebar.select_slider("Morphology kernel", options=[1,3,5,7,9,11,15], value=5),
    "iterations": st.sidebar.slider("Morphology iterations", 1, 5, 1),
    "adaptive_block": st.sidebar.select_slider("Adaptive block size", options=[3,5,7,9,11,15,21,31,51], value=11),
    "adaptive_c": st.sidebar.slider("Adaptive C", -10, 20, 2),
    "frequency_radius": st.sidebar.slider("Frequency radius", 3, 150, 30),
    "noise_sigma": st.sidebar.slider("Gaussian noise standard deviation", 1, 80, 20),
    "noise_amount": st.sidebar.slider("Salt & pepper amount", 0.01, 0.30, 0.05, 0.01),
    "seed": st.sidebar.number_input("Noise random seed", min_value=0, max_value=99999, value=42),
    "hue_min": st.sidebar.slider("HSV hue minimum", 0, 179, 0),
    "hue_max": st.sidebar.slider("HSV hue maximum", 0, 179, 30),
    "perspective_inset": st.sidebar.slider("Perspective inset", 0, 100, 20),
    "translate_x": st.sidebar.slider("Translate X", -200, 200, 0, 10),
    "translate_y": st.sidebar.slider("Translate Y", -200, 200, 0, 10),
}

try:
    processed, extra, explanation = apply_operation(original, module, operation, p)
except Exception as e:
    st.error(f"Processing failed: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(original, use_container_width=True)
    st.caption(f"Shape: {original.shape} • dtype: {original.dtype}")
with col2:
    st.subheader(f"Result: {operation}")
    st.image(as_displayable(processed), use_container_width=True)
    st.caption(f"Shape: {np.asarray(processed).shape} • dtype: {np.asarray(processed).dtype}")

if extra:
    st.info(extra)

st.markdown("### How it works")
st.write(explanation)
with st.expander("Parameter notes and practical guidance"):
    st.markdown("""
    - **Kernel size:** larger neighborhoods produce stronger smoothing or morphology effects. Odd sizes are used for centered kernels.
    - **Sigma:** controls Gaussian blur scale; higher values spread smoothing over a wider neighborhood.
    - **Thresholds:** choose values based on image intensity, noise, illumination, and the task.
    - **CLAHE:** clip limit controls local contrast amplification; tile size controls the local region scale.
    - **Frequency radius:** controls the size of the low-frequency region retained or removed in Fourier space.
    - **Medical images:** preserve the original, record every preprocessing step, and validate that subtle clinical structures are not removed. This app is for education, not diagnosis.
    """)

st.markdown("### Intensity histogram")
h1, h2 = st.columns(2)
with h1:
    show_histogram(gray(original), "Original grayscale intensity")
with h2:
    show_histogram(processed if np.asarray(processed).ndim == 2 else gray(as_displayable(processed)), "Processed grayscale intensity")

st.markdown("### Download")
d1, d2 = st.columns(2)
with d1:
    st.download_button("⬇️ Download processed image (PNG)", data=image_bytes(processed), file_name=f"processed_{operation.lower().replace(' ','_')}.png", mime="image/png")
with d2:
    st.download_button("⬇️ Download original image (PNG)", data=image_bytes(original), file_name="original.png", mime="image/png")

with st.expander("Tutorial: a recommended real-world workflow"):
    st.markdown("""
    1. **Inspect the data:** verify image dimensions, channels, bit depth, orientation, and file integrity.
    2. **Understand the task:** classification, detection, segmentation, OCR, enhancement, or measurement.
    3. **Establish a baseline:** start with the original image and only the preprocessing required by the model.
    4. **Apply one transformation at a time:** record parameters and compare outputs.
    5. **Check for data leakage:** split by patient, subject, video, or acquisition source before creating augmented derivatives.
    6. **Validate scientifically:** use domain-specific metrics and visual review. More visually dramatic does not always mean better.
    7. **Reproducibility:** keep a record of library versions, parameters, random seeds, and dataset provenance.
    """)
st.caption("Built with Streamlit, OpenCV, NumPy, Pillow, SciPy, and Matplotlib.")
