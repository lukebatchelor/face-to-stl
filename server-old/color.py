import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from skimage import color


def to_palette_str(palette):
    return ["{:02x}{:02x}{:02x}".format(int(r), int(g), int(b)) for r, g, b in palette]


def to_lightness(palette):
    return color.rgb2lab(palette.reshape(1, -1, 3)).reshape(-1, 3)


def sort_palette_by_brightness(palette):
    # Convert to grayscale
    gray = np.dot(palette, [0.299, 0.587, 0.114])

    # Sort by grayscale (brightness)
    sorted_idxs = np.argsort(gray)
    sorted_palette = palette[sorted_idxs]

    return sorted_palette


def quantize_colors(image, selected_colors, remap_colors_to_palette=True, reverse_palette=False):
    # Convert image to numpy array
    np_image = np.array(image)

    # Reshape the image to be a list of pixels
    pixels = np_image.reshape((-1, 3))

    # Perform k-means clustering
    n_colors = len(selected_colors)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    # Get the colors
    quantized_colors = kmeans.cluster_centers_

    # Sort quantized colors by brightness
    sorted_quantized_colors = sort_palette_by_brightness(quantized_colors)

    # Sort selected colors by brightness
    selected_colors = np.array(selected_colors)
    sorted_selected_colors = sort_palette_by_brightness(selected_colors)

    if reverse_palette:
        sorted_quantized_colors = sorted_quantized_colors[::-1]
        sorted_selected_colors = sorted_selected_colors[::-1]

    if remap_colors_to_palette:
        # Create a mapping from quantized colors to selected colors
        color_map = {
            tuple(q_color): tuple(s_color)
            for q_color, s_color in zip(sorted_quantized_colors, sorted_selected_colors)
        }

        # Apply the color mapping to the image
        labels = kmeans.labels_
        new_pixels = np.array(
            [color_map[tuple(quantized_colors[label])] for label in labels],
            dtype=np.uint8,
        )
    else:
        # Use the quantized colors directly
        new_pixels = quantized_colors[kmeans.labels_].astype(np.uint8)

    # Reshape back to the original image shape
    quantized_image = new_pixels.reshape(np_image.shape)

    # Convert back to PIL Image
    quantized_pil = Image.fromarray(quantized_image)

    # Get the color palette (sorted selected colors if remapped, otherwise quantized colors)
    if remap_colors_to_palette:
        color_palette = [
            "{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
            for r, g, b in sorted_selected_colors
        ]
    else:
        color_palette = [
            "{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
            for r, g, b in sorted_quantized_colors
        ]

    return quantized_pil, color_palette
