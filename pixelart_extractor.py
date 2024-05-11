'''
Module Name: pixelart_extractor

Description:
The PixelArt extractor analyses an image using various forms of image processing techniques.

A summary report can optionally be created.
Author: Donitz
License: MIT
Repository: https://github.com/Donitzo/ai-pixelart-extractor
'''

import numpy as np
import struct
import warnings

from numba import jit
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_prominences
from skimage.color import deltaE_cie76, lab2rgb, rgb2gray, rgb2lab
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.restoration import denoise_wavelet
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KDTree
from types import SimpleNamespace

def create_color_quantizer(image_lab:np.array, transparency_color_lab:np.array,
    same_color_cie76_threshold:float=10.0, max_colors:int=256):
    '''
    Creates a color palette for an image using K-Means clustering in the LAB color space.

    The color count will be reduced by removing similar colors until no two colors are within the specified threshold.

    :param np.array image_lab: The LAB representation of the image.
    :param np.array transparency_color_lab: The LAB representation of the transparency color.
    :param float same_color_cie76_threshold: The threshold for considering colors the same.
    :param int max_colors: The maximum number of colors in the final palette.
    :return: A tuple containing the new color palette in LAB space and a function used to apply it to an image in LAB space.
    :rtype: tuple(np.array, function)
    '''

    # Get unique colors from image
    pixels_lab = np.unique(image_lab.reshape(-1, 3), axis=0)

    # Remove transparent pixels
    delta = deltaE_cie76(pixels_lab, transparency_color_lab)
    is_opaque = delta > same_color_cie76_threshold
    pixels_lab = pixels_lab[is_opaque]

    if pixels_lab.shape[0] < 3:
        # Not enough colors for clustering
        palette_lab = pixels_lab
    else:
        # Generate an initial color palette using K-Means clustering
        n_clusters = min(max_colors - 1, pixels_lab.shape[0])
        kmeans_max = KMeans(n_clusters=n_clusters, random_state=42)
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
            kmeans_max.fit(pixels_lab)
        palette_lab = kmeans_max.cluster_centers_

    # Merge similar colors until the color differences are under the CIE76 threshold
    while palette_lab.shape[0] > 2:
        # Calculate the perceptual distance between every pair of colors
        a = np.tile(palette_lab.T, palette_lab.shape[0]).T
        b = np.repeat(palette_lab, palette_lab.shape[0], axis=0)
        deltas = deltaE_cie76(a, b)
        deltas[deltas == 0] = 1e6 # But not itself

        # Stop if all colors are different enough
        if deltas.min() > same_color_cie76_threshold:
            break

        # Erase the (less common) color with the smaller cluster
        i_a = deltas.argmin() % palette_lab.shape[0]
        i_b = int(deltas.argmin() / palette_lab.shape[0])
        palette_lab = palette_lab[np.arange(palette_lab.shape[0]) != max(i_a, i_b)]

    # Add the transparency color to the palette
    palette_lab = np.vstack([palette_lab, [transparency_color_lab]])

    # Create a lookup tree for the new color palette
    kdtree = KDTree(palette_lab, leaf_size=8)

    def apply_palette_lab(image_lab:np.array):
        pixels_lab = palette_lab[kdtree.query(image_lab.reshape(-1, 3))[1]]
        is_opaque = deltaE_cie76(pixels_lab, transparency_color_lab) > same_color_cie76_threshold
        pixels_lab[~is_opaque] = transparency_color_lab
        return pixels_lab.reshape(image_lab.shape)

    return palette_lab, apply_palette_lab

def find_common_edge_color(image_lab:np.array, same_color_cie76_threshold:float=10, depth:int|None=None):
    '''
    Finds the most common color along the edges of an image within the specified CIE76 threshold.

    :param np.array image_lab: The LAB representation of the image.
    :param float same_color_cie76_threshold: The threshold for considering colors the same.
    :param int depth: The depth of edge in pixels (default is 2% of the image height).
    :return: A tuple with the mean of the found color and what fraction of the edges included this color
    :rtype: (np.array, float)
    '''

    if depth is None:
        depth = int(image_lab.shape[0] * 0.02) + 1

    # Get pixels around the edges of image
    edge_colors = np.vstack([
        image_lab[:depth, :, :].reshape(-1, 3),
        image_lab[-depth:, :, :].reshape(-1, 3),
        image_lab[:, :depth, :].reshape(-1, 3),
        image_lab[:, -depth:, :].reshape(-1, 3),
    ])
    edge_count = edge_colors.shape[0]

    # Merge and count edge colors

    mean_color = None
    largest_fraction = 0

    while edge_colors.shape[0] > 0:
        is_similar = deltaE_cie76(edge_colors[0], edge_colors) < same_color_cie76_threshold
        fraction = is_similar.sum() / float(edge_count)
        if fraction > largest_fraction:
            largest_fraction = fraction
            mean_color = edge_colors[is_similar].mean(axis=0)
        edge_colors = edge_colors[~is_similar]

    return mean_color, largest_fraction

def alpha_to_transparency_color(image_rgba:np.array, transparency_color_rgb:np.array, alpha_threshold:float=0.5):
    '''
    Converts pixels with low alpha values in an RGBA image to the specified transparency color.

    :param np.array image: The input RGBA/RGB image.
    :param np.array transparency_color_rgb: The RGB transparency color.
    :param float alpha_threshold: The alpha threshold for considering pixels as transparent.
    :return: The modified RGB image with pixels converted to the transparency color.
    :rtype: np.array
    '''

    if image_rgba.shape[-1] == 4:
        is_transparent = image_rgba[:, :, 3] <= alpha_threshold
        image_rgba[is_transparent, :3] = transparency_color_rgb

    return image_rgba[:, :, :3]

def find_symmetrical_x_center(image_rgb:np.array):
    '''
    Finds the center of symmetry along the horizontal axis in an RGB image.

    :param np.array image_rgb: The input RGB image.
    :return: The index of the symmetry center and the mean correlation coefficient.
    :rtype: tuple(int, float)
    '''

    grayscale = rgb2gray(image_rgb)

    # Correlate each horizontal row in the image with the reversed row

    correlation = np.zeros(grayscale.shape[1], dtype=np.float32)

    for i in range(grayscale.shape[0]):
        x = grayscale[i, :]
        y = np.flip(x)
        x_d = np.std(x) * x.shape[0]
        y_d = np.std(y)
        x = np.divide(x - np.mean(x), x_d, np.zeros_like(x), where=x_d != 0)
        y = np.divide(y - np.mean(y), y_d, np.zeros_like(y), where=y_d != 0)
        correlation += np.correlate(x, y, mode='same')

    correlation /= grayscale.shape[0]

    # Find the symmetry center and correlation coefficient
    i_max = correlation.argmax()
    r_max = correlation[i_max]

    return i_max, r_max

def pad_x_to_center(image:np.array, target_x:float):
    '''
    Pads an image along the horizontal axis to center it around the specified x-coordinate.

    :param np.array image: The input image.
    :param float target_x: The x-coordinate around which the image should be centered.
    :return: The padded image.
    :rtype: np.array
    '''

    # Calculate the required shift from the center
    current_center = image.shape[1] / 2.0
    shift_needed = int(target_x - current_center)

    # Calculate the required padding
    padding_left = max(0, shift_needed)
    padding_right = max(0, -shift_needed)

    # Pad image
    if padding_left + padding_right == 0:
        return image
    else:
        return np.pad(image, ((0, 0), (padding_right, padding_left), (0, 0)), mode='constant', constant_values=0)

def crop_color(image_lab:np.array, color_lab:np.array, same_color_cie76_threshold:float=10, padding:int=5):
    '''
    Crops an image around the specified color, within a given color difference threshold.

    :param np.array image_lab: The LAB representation of the image.
    :param np.array color_lab: The LAB representation of the color to crop around.
    :param float same_color_cie76_threshold: The threshold when comparing the color.
    :param int padding: The padding to add around the cropped region.
    :return: The cropped region of the image.
    :rtype: np.array
    '''

    delta = deltaE_cie76(image_lab, color_lab)
    keep = delta > same_color_cie76_threshold

    indices = np.argwhere(keep)
    if indices.shape[0] == 0:
        return image_lab

    top_left = indices.min(axis=0)
    bottom_right = indices.max(axis=0)

    top = max(top_left[0] - padding, 0)
    left = max(top_left[1] - padding, 0)
    bottom = min(bottom_right[0] + padding + 1, image_lab.shape[0])
    right = min(bottom_right[1] + padding + 1, image_lab.shape[1])

    return image_lab[top:bottom, left:right]

def create_edge_profile(image_lab:np.array, horizontal:bool):
    '''
    Create an edge profile for an image along the specified axis.
    The profile indicates where and how intense CIE76 color changes are.

    :param np.array image_lab: The LAB representation of the image.
    :param bool horizontal: If True, analyzes horizontal edges; if False, analyzes vertical edges.
    :return: A tuple containing an image with the CIE76 delta and the 1D edge profile.
    :rtype: tuple(np.array, np.array)
    '''

    if not horizontal:
        image_lab = np.swapaxes(image_lab, 0, 1)

    # Calculate the perceptual color difference using CIE76
    delta = np.abs(deltaE_cie76(image_lab[:, :-1], image_lab[:, 1:], channel_axis=2)) > 0
    delta = np.hstack([delta, np.zeros((delta.shape[0], 1))])

    # Sum edges along the vertical axis and normalize
    edge_profile = delta.mean(0)
    if edge_profile.max() > 0:
        edge_profile /= edge_profile.max()

    return delta, edge_profile

@jit(nopython=True)
def get_spacing_error(spacing:float, sorted_x:np.ndarray, min_x:float, max_x:float, gap_penalty_weight:float=1.0) -> float:
    '''
    Compute the spacing error based on two metrics:

    1. The normalized RMSE between the peaks and the expected gap between peaks.
    2. The normalized number of missing peaks * gap_penalty_weight.

    :param float spacing: The pixel spacing to test.
    :param np.array sorted_x: The peak coordinates sorted by X.
    :param float min_x: The minimum X coordinate of the image.
    :param float max_x: The maximum X coordinate of the image.
    :param float gap_penalty_weight: How much to penalize gaps (missing points).
    :return: The combined error.
    :rtype: float
    '''

    # Calculate spacing error and count gaps

    gaps = 0
    total_expected_points = 0
    squared_error = 0.0
    n = len(sorted_x) - 1

    for i in range(n):
        actual_distance = sorted_x[i + 1] - sorted_x[i]
        expected_points = max(1, round(actual_distance / spacing))
        expected_distance = expected_points * spacing
        squared_error += (actual_distance - expected_distance) ** 2
        gaps += max(0, expected_points - 1)
        total_expected_points += expected_points

    # Add additional missing points outside the peak range
    missing_points = round((max(0, sorted_x.min() - min_x) + max(0, max_x - sorted_x.max())) / spacing)
    gaps += missing_points
    total_expected_points += missing_points

    # Calculate a penalty based on the number of gaps to fill in
    gap_penalty = 0 if total_expected_points == 0 else\
        gaps / total_expected_points * gap_penalty_weight

    # Normalize RMSE
    rmse = np.sqrt(squared_error / n)
    normalized_error = rmse / spacing

    return normalized_error + gap_penalty

def find_optimal_spacing(peaks:np.ndarray, prominences:np.ndarray, min_peaks:int, min_x:float,
    max_x:float, smallest_spacing:float=1.0, largest_spacing:float=64.0, gap_penalty_weight:float=1.0):
    '''
    Find the optimal grid spacing using an optimization method. See get_spacing_error.

    :param np.array peaks: The X peaks of edge profile.
    :param np.array prominences: The prominence of the X peaks.
    :param int min_peaks: The minimum number of peaks to consider.
    :param float min_x: The minimum X coordinate of the image.
    :param float max_x: The maximum X coordinate of the image.
    :param float smallest_spacing: The lower spacing bound.
    :param float largest_spacing: The upper spacing bound.
    :param float gap_penalty_weight: How much to penalize gaps (missing points).
    :return: A tuple with arrays of spacings, errors, peak_counts.
    :rtype: tuple(np.array, np.array, np.array)
    '''

    # Sort peaks
    indices = np.argsort(-prominences)
    peaks = peaks[indices]
    prominences = prominences[indices]

    # Calculate the spacing and error for all possible peak counts (in increasing prominence),

    peak_counts = np.arange(min_peaks, peaks.shape[0])
    errors = np.zeros_like(peak_counts, dtype=np.float32)
    spacings = np.zeros_like(peak_counts, dtype=np.float32)

    for i, peak_count in enumerate(peak_counts):
        sorted_x = np.sort(peaks[:peak_count])

        initial_guess = np.median(np.diff(sorted_x))

        result = minimize(lambda spacing: get_spacing_error(spacing[0], sorted_x, min_x, max_x, gap_penalty_weight),
            [initial_guess], bounds=[(smallest_spacing, largest_spacing)], method='L-BFGS-B')

        spacings[i] = result.x[0]
        errors[i] = get_spacing_error(result.x[0], sorted_x, min_x, max_x, gap_penalty_weight)

    return spacings, errors, peak_counts

@jit(nopython=True)
def find_edges(peaks:np.ndarray, prominences:np.ndarray,
    spacing:float, cell_trim_fraction:float=0.7, gap_fraction:float=1.5) -> np.ndarray:
    '''
    Find the pixel edges from X peaks using the specified spacing.

    This method iterates the peaks in order of prominence,
    and removes lower peaks within cell_trim_fraction.
    The remaining peaks are padded with missing edges.

    :param np.array peaks: The X peaks of edge profile.
    :param np.array prominences: The prominence of the X peaks.
    :param float spacing: The spacing (pixel size).
    :param float cell_trim_fraction: What fraction of spacing to trim around peaks.
    :param float gap_fraction: What fraction of gap spacing to add edges to.
    :return: An array of the pixel edges.
    :rtype: np.array
    '''

    # Sort peaks
    indices = np.argsort(-prominences)
    peaks = peaks[indices]
    prominences = prominences[indices]

    # Trim smaller peaks around the most prominent peaks

    distance_threshold = spacing * cell_trim_fraction

    filtered_peaks = []
    filtered_prominences = []

    for i in range(len(peaks)):
        x = peaks[i]
        prominence = prominences[i]
        should_add = True

        for j in range(len(filtered_peaks)):
            f_x = filtered_peaks[j]
            f_prominence = filtered_prominences[j]

            if not (x == f_x or abs(x - f_x) > distance_threshold or prominence > f_prominence):
                should_add = False
                break

        if should_add:
            filtered_peaks.append(x)
            filtered_prominences.append(prominence)

    filtered_peaks = np.sort(np.array(filtered_peaks))

    # Insert missing edges in gaps

    gap_threshold = spacing * gap_fraction

    edges = [filtered_peaks[0] - spacing]

    for i in range(len(filtered_peaks)):
        edges.append(filtered_peaks[i])
        if i < len(filtered_peaks) - 1:
            gap = filtered_peaks[i + 1] - filtered_peaks[i]
            if gap > gap_threshold:
                divisions = int(round(gap / spacing) - 1)
                for n in range(divisions):
                    edges.append(edges[-1] + gap / (divisions + 1))

    edges.append(filtered_peaks[-1] + spacing)

    return np.array(edges, dtype=np.float32) + 0.5

def sample_pixels(image_lab:np.array, transparency_color_lab:np.array,
    pixel_edges_x:np.array, pixel_edges_y:np.array, pixel_padding_fraction:float=0.25, sampling_grid_size:int=8):
    '''
    Samples the center of pixels from an image based on the pixel edges.

    :param np.array image_lab: The LAB representation of the image.
    :param np.array transparency_color_lab: The LAB representation of the transparency color.
    :param np.array pixel_edges_x: The edges of pixels along the x-axis.
    :param np.array pixel_edges_y: The edges of pixels along the y-axis.
    :return: The sampled image in LAB space.
    :rtype: np.array
    '''

    x = np.arange(image_lab.shape[1])
    y = np.arange(image_lab.shape[0])
    interpolator = RegularGridInterpolator((y, x), image_lab,
        method='nearest', bounds_error=False, fill_value=transparency_color_lab)

    sample_x = pixel_edges_x[:-1] + np.diff(pixel_edges_x) * 0.5
    sample_y = pixel_edges_y[:-1] + np.diff(pixel_edges_y) * 0.5
    sample_points = np.array(np.meshgrid(sample_y, sample_x)).T.reshape(-1, 2)

    return interpolator(sample_points).reshape(pixel_edges_y.shape[0] - 1, pixel_edges_x.shape[0] - 1, 3)

def remove_isolated_small_objects(image_rgba:np.array, land_dilution:int=2, island_size:int=3):
    '''
    Removes small, isolated objects that do not connect to larger objects within a specified dilution range.

    :param np.array image_rgba: The input RGBA image.
    :param int land_dilution: The radius of dilation applied to remaining objects after initial removal.
    :param int island_size: The minimum size of objects to remove.
    :return: The cleaned RGBA image.
    :rtype: np.array
    '''

    opaque = image_rgba[:, :, 3] > 0.5
    cleaned = remove_small_objects(opaque, min_size=island_size)
    dilated = binary_dilation(cleaned, disk(land_dilution))
    diluted_with_islands = dilated | opaque
    final_cleaned = remove_small_objects(diluted_with_islands, min_size=island_size)
    remove_mask = opaque & ~final_cleaned

    for i in range(4):
        image_rgba[remove_mask, i] = 0

    return image_rgba

def split_image(image_lab:np.array, transparency_color_lab:np.array, same_color_cie76_threshold:float=10, min_distance:int=10):
    '''
    Splits an image into subregions disconnected by at least the min distance.

    :param np.array image_lab: The LAB representation of the image.
    :param np.array transparency_color_lab: The LAB representation of the transparency color.
    :param float same_color_cie76_threshold: The threshold for considering colors the same.
    :param int min_distance: The minimum pixel distance.
    :return: A list of LAB images.
    :rtype: [np.array]
    '''

    delta = deltaE_cie76(image_lab, transparency_color_lab)
    is_opaque = delta > same_color_cie76_threshold

    dilated = binary_dilation(is_opaque, disk(min_distance))
    labeled_image = label(dilated)
    regions = regionprops(labeled_image)

    images = []

    for region in regions:
        min_y, min_x, max_y, max_x = region.bbox
        images.append(image_lab[min_y:max_y, min_x:max_x, :])

    return images

def extract_sprites(image_rgba:np.array, detect_transparency_color:bool=True, default_transparency_color_hex:str='ff00ff',
    split_distance:int|None=None, min_sprite_size:int=8, same_color_cie76_threshold:float=10.0, border_transparency_cie76_threshold:float=20,
    max_colors:int=256, largest_pixel_size:int=64, minimum_peak_fraction:float=0.2, land_dilution_during_cleanup:int=1,
    island_size_to_remove:int=5, symmetry_coefficient_threshold:float=0.5, create_summary:bool=False):
    '''
    Extracts a sprite from an image by applying various image processing techniques.
    The function handles transparency, denoising, color quantization, edge detection,
    peak finding, optimal spacing determination, symmetry finding, and visualization.

    :param np.array image_rgb: The input image.
    :param bool detect_transparency_color: Whether to automatically detect the transparency color.
    :param str default_transparency_color_hex: The default color to use for transparency.
    :param float same_color_cie76_threshold: The threshold for considering colors the same.
    :param float border_transparency_cie76_threshold: The threshold used to detect/refine transparent borders.
    :param int | None split_distance: If set, this distance is used to split the image into multiple separated sprites.
    :param int min_sprite_size: Sprites smaller than this are discarded.
    :param int max_colors: The maximum number of colors in the final palette.
    :param int largest_pixel_size: The largest pixel size in the unscaled image.
    :param float minimum_peak_fraction: The minimum fraction of peaks to consider during edge detection.
    :param int land_dilution_during_cleanup: The radius of dilation applied when removing islands.
    :param int island_size_to_remove: The minimum size of objects to remove.
    :param float symmetry_coefficient_threshold: The mean correlation coefficient required for symmetry detection.
    :param bool create_summary: Whether to create a summary image.
    :return: A list of SingleNamespaces with the sprite, whether it was centered and an optional summary image. A None element is returned on failure.
    :rtype: list
    '''

    print(f'Extracting sprite from image of size {image_rgba.shape[1]}x{image_rgba.shape[0]}')

    # Always work in the color range 0-1
    image_rgba = image_rgba / 255.0

    # The default transparency color is used as a placeholder for the alpha channel

    color_bytes = struct.unpack('BBB', bytes.fromhex(default_transparency_color_hex))
    transparency_color_rgb = np.array(list(map(lambda x: x / 255.0, color_bytes)), dtype=np.float32)
    transparency_color_lab = rgb2lab(transparency_color_rgb)

    image_rgb = alpha_to_transparency_color(image_rgba, transparency_color_rgb)
    image_lab = rgb2lab(image_rgb)

    # De-noise the image using Wavelet denoising
    with warnings.catch_warnings():
        warnings.filterwarnings('error', r'(Mean of empty slice.)|(Level value of 1 is too high: all coefficients will experience boundary effects.)')
        try:
            image_rgb = denoise_wavelet(image_rgb, convert2ycbcr=True, rescale_sigma=True, channel_axis=2)
        except Warning:
            print('Problem running wavelet denoising. Skipping...')

    # Automatically detect the transparency color

    REQUIRED_EDGE_FRACTION = 0.75

    if detect_transparency_color:
        color, fraction = find_common_edge_color(image_lab, same_color_cie76_threshold)

        if fraction > REQUIRED_EDGE_FRACTION:
            transparency_color_lab = color
            transparency_color_rgb = lab2rgb(color)

            print(f'Transparency color {transparency_color_rgb} found ({fraction * 100:.1f} % of image edges)')

    # Split into subregions
    if split_distance is None:
        images_lab = [image_lab]
    else:
        images_lab = split_image(image_lab, transparency_color_lab, min_distance=split_distance)

    print(f'Split image into {len(images_lab)} subregions')

    # Process each subregion

    results = []

    for image_index, image_lab in enumerate(images_lab):
        print(f'Processing subregion {image_index + 1} of {len(images_lab)}')

        # Crop transparent areas
        image_lab = crop_color(image_lab, transparency_color_lab)

        print(f'Cropped image to size {image_lab.shape[1]}x{image_lab.shape[0]}')

        # Quantize the full image colors
        _, quantizer_lab_full = create_color_quantizer(
            image_lab=image_lab,
            transparency_color_lab=transparency_color_lab,
            same_color_cie76_threshold=same_color_cie76_threshold,
            max_colors=max_colors
        )

        indexed_image_lab = quantizer_lab_full(image_lab)

        # Find the edges between colors
        image_edges_x, profile_x = create_edge_profile(indexed_image_lab, True)
        image_edges_y, profile_y = create_edge_profile(indexed_image_lab, False)

        # Find the profile peaks
        peaks_x = find_peaks(profile_x)[0]
        peaks_y = find_peaks(profile_y)[0]
        prominences_x = peak_prominences(profile_x, peaks_x)[0]
        prominences_y = peak_prominences(profile_y, peaks_y)[0]

        # Not enough peaks in the image
        if prominences_x.shape[0] < 5 or prominences_y.shape[0] < 5:
            print('Sprite is too small or uniform')
            results.append(None)
            continue

        # Find the pixel size

        min_peaks_x = max(3, int(peaks_x.shape[0] * minimum_peak_fraction))
        min_peaks_y = max(3, int(peaks_y.shape[0] * minimum_peak_fraction))

        spacings_x, errors_x, peak_counts_x = find_optimal_spacing(peaks_x, prominences_x, min_peaks_x,
            0, image_edges_x.shape[0], largest_spacing=largest_pixel_size)
        spacings_y, errors_y, peak_counts_y = find_optimal_spacing(peaks_y, prominences_y, min_peaks_y,
            0, image_edges_y.shape[0], largest_spacing=largest_pixel_size)

        best_index_x = errors_x.argmin()
        best_index_y = errors_y.argmin()

        spacing_x = spacings_x[best_index_x]
        spacing_y = spacings_y[best_index_y]

        error_x = errors_x[best_index_x]
        error_y = errors_y[best_index_y]

        # Find the edges
        edges_x = find_edges(peaks_x, prominences_x, spacing_x)
        edges_y = find_edges(peaks_y, prominences_y, spacing_y)

        # Too small?
        if edges_x.shape[0] < min_sprite_size and edges_y.shape[0] < min_sprite_size:
            print('The resulting sprite is too small')
            results.append(None)
            continue

        # Generate a new palette for the extracted sprite
        sprite_lab = sample_pixels(image_lab, transparency_color_lab, edges_x, edges_y)

        palette_lab, quantizer_lab = create_color_quantizer(
            image_lab=sprite_lab,
            transparency_color_lab=transparency_color_lab,
            same_color_cie76_threshold=same_color_cie76_threshold,
            max_colors=max_colors
        )

        indexed_sprite_lab = quantizer_lab(sprite_lab)

        print(f'Sprite extracted of size {indexed_sprite_lab.shape[1]}x{indexed_sprite_lab.shape[0]} with {palette_lab.shape[0]} colors')

        # Find sprite symmetry
        center_x, x_r = find_symmetrical_x_center(lab2rgb(indexed_sprite_lab))
        center_y, y_r = find_symmetrical_x_center(lab2rgb(indexed_sprite_lab).swapaxes(0, 1))

        # Remove transparency (and semi-transparent borders)
        delta = deltaE_cie76(sprite_lab, transparency_color_lab)
        is_transparent = delta < same_color_cie76_threshold
        diluted_transparency = binary_dilation(is_transparent, disk(1))
        diluted_transparency = is_transparent | (diluted_transparency & (delta < border_transparency_cie76_threshold))
        alpha = np.where(diluted_transparency, 0, 1)

        # Add the alpha channel back to the sprite
        indexed_sprite_rgba = np.dstack((lab2rgb(indexed_sprite_lab), alpha))
        indexed_sprite_rgba[alpha == 0] = 0

        # Cleanup small islands of pixels
        if island_size_to_remove > 0:
            indexed_sprite_rgba = remove_isolated_small_objects(indexed_sprite_rgba, land_dilution=land_dilution_during_cleanup, island_size=island_size_to_remove)

        # If the symmetry correlation coefficients are high enough, center the symmetry center in either axis
        centered_x = x_r > symmetry_coefficient_threshold
        centered_y = y_r > symmetry_coefficient_threshold
        if centered_x:
            indexed_sprite_rgba = pad_x_to_center(indexed_sprite_rgba, center_x)
            print(f'X symmetry found. Centered sprite horizontally.')
        if centered_y:
            indexed_sprite_rgba = pad_x_to_center(indexed_sprite_rgba.swapaxes(0, 1), center_y).swapaxes(0, 1)
            print(f'Y symmetry found. Centered sprite vertically.')

        # Convert sprite to bytes
        indexed_sprite_rgba = np.rint(indexed_sprite_rgba * 255).astype(np.uint8)

        # Create summary plot
        if create_summary:
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt

            plt.style.use('dark_background')
            fig = plt.figure(figsize=(24, 15))

            ax = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2, fig=fig)
            ax.imshow(np.dstack([image_edges_x, image_edges_y.T, image_edges_x * 0]))
            ax.set_title('Horizontal/Vertical color deltas')

            ax = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=2, fig=fig)
            ax.imshow(lab2rgb(indexed_image_lab))
            sample_points = np.array(np.meshgrid((edges_x[1:] + edges_x[:-1]) / 2.0, (edges_y[1:] + edges_y[:-1]) / 2.0)).T.reshape(-1, 2)
            ax.scatter(sample_points[:, 0], sample_points[:, 1], s=0.3, c='yellow')
            ax.vlines(edges_x, ymin=edges_y.min(), ymax=edges_y.max(), alpha=0.4, colors='yellow')
            ax.hlines(edges_y, xmin=edges_x.min(), xmax=edges_x.max(), alpha=0.4, colors='yellow')
            ax.set_title(f'Pixel edges and sample points')

            ax = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2, fig=fig)
            ax.imshow(indexed_sprite_rgba)
            if centered_x:
                ax.vlines(indexed_sprite_rgba.shape[1] / 2.0 - 0.5, ymin=0, ymax=indexed_sprite_rgba.shape[0] - 1, alpha=0.4, colors='red', label='Center X')
            if centered_y:
                ax.hlines(indexed_sprite_rgba.shape[0] / 2.0 - 0.5, xmin=0, xmax=indexed_sprite_rgba.shape[1] - 1, alpha=0.4, colors='red', label='Center Y')
            ax.set_title(f'Extracted {indexed_sprite_rgba.shape[1]}x{indexed_sprite_rgba.shape[0]} sprite with {palette_lab.shape[0]} colors')

            ax = plt.subplot2grid((4, 6), (2, 0), colspan=3, fig=fig)
            ax.plot(peak_counts_x, spacings_x, color='yellow', label='Pixel size')
            ax.vlines(peak_counts_x[best_index_x], ymin=0, ymax=spacings_x.max(), alpha=0.4, colors='green', label='Best size X')
            ax.set_xlabel('Peak count (sorted by prominence)')
            ax.set_ylabel('Pixel size')
            ax.set_title(f'Pixel size estimation for X axis (pixel size {spacing_x:.2f} with error {error_x:.2f} from {peak_counts_x[best_index_x]} peaks)')

            ax2 = ax.twinx()
            ax2.plot(peak_counts_x, errors_x, color='red', label='Spacing/gap error')
            ax2.set_ylabel('Error')
            plots_1, labels_1 = ax.get_legend_handles_labels()
            plots_2, labels_2 = ax2.get_legend_handles_labels()
            ax2.legend(plots_1 + plots_2, labels_1 + labels_2, loc='upper right')

            ax = plt.subplot2grid((4, 6), (2, 3), colspan=3, fig=fig)
            ax.plot(peak_counts_y, spacings_y, color='yellow', label='Pixel size')
            ax.vlines(peak_counts_y[best_index_y], ymin=0, ymax=spacings_y.max(), alpha=0.4, colors='green', label='Best size Y')
            ax.set_xlabel('Peak count (sorted by prominence)')
            ax.set_ylabel('Pixel size')
            ax.set_title(f'Pixel size estimation for Y axis (pixel size {spacing_y:.2f} with error {error_y:.2f} from {peak_counts_y[best_index_y]} peaks)')

            ax2 = ax.twinx()
            ax2.plot(peak_counts_y, errors_y, color='red', label='Spacing/gap error')
            ax2.set_ylabel('Error')
            plots_1, labels_1 = ax.get_legend_handles_labels()
            plots_2, labels_2 = ax2.get_legend_handles_labels()
            ax2.legend(plots_1 + plots_2, labels_1 + labels_2, loc='upper right')

            ax = plt.subplot2grid((4, 6), (3, 0), colspan=3, fig=fig)
            ax.plot(np.arange(profile_x.shape[0]) + 0.5, profile_x, color='yellow', label='Edge profile X')
            ax.vlines(edges_x, ymin=0, ymax=profile_x.max(), alpha=0.4, colors='red', label='Found pixel edges')
            ax.set_xlabel('X')
            ax.set_ylabel('Edge intensity')
            ax.set_title(f'Edge profile X and found pixel edges')
            ax.legend(loc='upper right')

            ax = plt.subplot2grid((4, 6), (3, 3), colspan=3, fig=fig)
            ax.plot(np.arange(profile_y.shape[0]) + 0.5, profile_y, color='yellow', label='Edge profile Y')
            ax.vlines(edges_y, ymin=0, ymax=profile_y.max(), alpha=0.4, colors='red', label='Found pixel edges')
            ax.set_xlabel('X')
            ax.set_ylabel('Edge intensity')
            ax.set_title('Edge profile Y and found pixel edges')
            ax.legend(loc='upper right')

            fig.tight_layout()
            fig.canvas.draw()

            summary_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)\
                .reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close()
        else:
            summary_rgb = None

        results.append(SimpleNamespace(
            sprite_rgba=indexed_sprite_rgba,
            centered_x=centered_x,
            centered_y=centered_y,
            pixel_size_x=spacing_x,
            pixel_size_y=spacing_y,
            summary_rgb=summary_rgb))

    return results
