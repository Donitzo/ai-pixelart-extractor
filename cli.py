'''
Module Name: cli

Description:
Command-line utility for the AI PixelArt Extractor.

Author: Donitz
License: MIT
Repository: https://github.com/Donitzo/ai-pixelart-extractor
'''

import argparse
import glob
import os
import sys

from skimage import io

from pixelart_extractor import extract_sprites

class UnsupportedVersion(Exception):
    pass

MIN_VERSION, VERSION_LESS_THAN = (3, 6), (4, 0)
if sys.version_info < MIN_VERSION or sys.version_info >= VERSION_LESS_THAN:
    raise UnsupportedVersion('requires Python %s,<%s' % ('.'.join(map(str, MIN_VERSION)), '.'.join(map(str, VERSION_LESS_THAN))))

parser = argparse.ArgumentParser()
parser.add_argument('--image_path_pattern', required=True)
parser.add_argument('--output_directory', default='output')
parser.add_argument('--summary_directory', default='summary')
parser.add_argument('--detect_transparency_color', action='store_false')
parser.add_argument('--transparency_color_hex', default='ff00ff')
parser.add_argument('--same_color_cie76_threshold', type=float, default=10.0)
parser.add_argument('--border_transparency_cie76_threshold', type=float, default=20.0)
parser.add_argument('--split_distance', type=int, default=0)
parser.add_argument('--min_sprite_size', type=int, default=8)
parser.add_argument('--max_colors', type=int, default=128)
parser.add_argument('--largest_pixel_size', type=int, default=64)
parser.add_argument('--minimum_peak_fraction', type=float, default=0.2)
parser.add_argument('--land_dilution_during_cleanup', type=int, default=1)
parser.add_argument('--island_size_to_remove', type=int, default=5)
parser.add_argument('--symmetry_coefficient_threshold', type=float, default=0.5)
parser.add_argument('--create_summary', action='store_true')
args = parser.parse_args()

image_paths = glob.glob(args.image_path_pattern)

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

if not os.path.exists(args.summary_directory):
    os.makedirs(args.summary_directory)

for image_path in image_paths:
    print(f'\nExtracting sprite from image "{image_path}"')

    base_path = os.path.splitext(image_path)[0]

    image = io.imread(image_path)

    # Handle GIF
    if len(image.shape) == 4:
        image = image[0]

    # Extract sprite
    sprites = extract_sprites(
        image,
        detect_transparency_color=args.detect_transparency_color,
        default_transparency_color_hex=args.transparency_color_hex,
        same_color_cie76_threshold=args.same_color_cie76_threshold,
        border_transparency_cie76_threshold=args.border_transparency_cie76_threshold,
        split_distance=None if args.split_distance == 0 else args.split_distance,
        min_sprite_size=args.min_sprite_size,
        max_colors=args.max_colors,
        largest_pixel_size=args.largest_pixel_size,
        minimum_peak_fraction=args.minimum_peak_fraction,
        land_dilution_during_cleanup=args.land_dilution_during_cleanup,
        island_size_to_remove=args.island_size_to_remove,
        symmetry_coefficient_threshold=args.symmetry_coefficient_threshold,
        create_summary=args.create_summary
    )

    count = 0

    for i, sprite in enumerate(sprites):
        if sprite is None:
            print(f'Failed to extract sprite {i}')
            continue

        if not sprite.summary_rgb is None:
            summary_path = os.path.join(args.summary_directory,
                f'{os.path.basename(base_path)}_{count}_summary.png')

            io.imsave(summary_path, sprite.summary_rgb)

        sprite_path = os.path.join(args.output_directory,
            f'{os.path.basename(base_path)}_{count}_extracted{"_cx" if sprite.centered_x else ""}{"_cy" if sprite.centered_y else ""}.png')

        io.imsave(sprite_path, sprite.sprite_rgba)

        count += 1

        print(f'Sprite saved to "{sprite_path}"')

print('\nDone')
