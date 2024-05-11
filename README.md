# AI Pixel-Art Extractor

A Work-in-progress. This pixel-art extractor uses various sophisticated image processing methods to try its best to extract machine generated and just badly aligned pixel-art.

The pixel-art extractor works with virtual pixel sizes down to 2 image. Any smaller and the art won't be properly detected. The extractor is not flawless and can often fail on very badly aliased or misaligned pixelart.

## Example usage

`python cli.py --image_path_pattern "./images" --split_distance 10 --create_summary`