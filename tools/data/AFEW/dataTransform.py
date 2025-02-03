# Transformation function for pixel to circumplex range (-10 to 10)
def transform_coordinates(pixel_x, pixel_y, origin_x=200, origin_y=200, graph_range=20, pixels_span=400):
    """
    Transforms pixel coordinates to the circumplex model's (-10, 10) range.

    Args:
        pixel_x, pixel_y: Pixel coordinates in the image.
        origin_x, origin_y: The pixel coordinates of the origin (center of the graph).
        graph_range: The numerical range of the graph (-10 to 10 = 20).
        pixels_span: The total span of the graph in pixels.

    Returns:
        (pleasure, arousal): Coordinates mapped to the circumplex range (-10, 10).
    """
    scale = graph_range / pixels_span  # Scale for converting pixels to circumplex range
    pleasure = (pixel_x - origin_x) * scale  # Horizontal coordinate
    arousal = -(pixel_y - origin_y) * scale  # Vertical coordinate (negative due to inverted y-axis)
    return pleasure, arousal


# Dictionary of approximate pixel coordinates for each affect category
affect_labels = {
    "Alarmed": (240, 80),
    "Afraid": (220, 100),
    "Angry": (210, 90),
    "Annoyed": (190, 120),
    "Distressed": (180, 150),
    "Frustrated": (170, 180),
    "Miserable": (100, 230),
    "Sad": (150, 250),
    "Gloomy": (130, 260),
    "Depressed": (160, 280),
    "Bored": (170, 310),
    "Droopy": (190, 330),
    "Tired": (200, 340),
    "Sleepy": (210, 350),
    "Aroused": (250, 80),
    "Astonished": (270, 90),
    "Excited": (290, 100),
    "Delighted": (320, 140),
    "Happy": (310, 190),
    "Pleased": (300, 200),
    "Glad": (280, 220),
    "Serene": (260, 280),
    "At ease": (250, 300),
    "Content": (240, 320),
    "Relaxed": (230, 330),
    "Satisfied": (220, 310),
    "Calm": (210, 300)
}

# Transform each affect category's pixel coordinates into the circumplex model's (-10, 10) range
results = {label: transform_coordinates(x, y) for label, (x, y) in affect_labels.items()}

# Print results
for label, (pleasure, arousal) in results.items():
    print(f"{label}: Pleasure = {pleasure:.2f}, Arousal = {arousal:.2f}")
