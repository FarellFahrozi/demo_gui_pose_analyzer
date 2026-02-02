
def create_rounded_rect(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    """
    Membuat rectangle dengan rounded corners pada canvas.
    
    Args:
        canvas: Tkinter Canvas object
        x1, y1, x2, y2: Koordinat bounding box
        radius: Radius sudut tumpul (default: 25)
        **kwargs: Argumen tambahan untuk create_polygon (fill, outline, etc.)
    """
    points = [
        x1+radius, y1,
        x2-radius, y1,
        x2, y1,
        x2, y1+radius,
        x2, y2-radius,
        x2, y2,
        x2-radius, y2,
        x1+radius, y2,
        x1, y2,
        x1, y2-radius,
        x1, y1+radius,
        x1, y1
    ]
    
    return canvas.create_polygon(points, **kwargs, smooth=True)
