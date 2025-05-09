import cv2
import numpy as np
import os
import glob 
from PIL import Image


def find_chessboard(image_path, debug=False):
    """
    Find and extract a chessboard from an image.
    Returns the warped top-down view of the chessboard.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not load image from {image_path}")
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding instead of simple blur
    # This helps with different lighting conditions
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Use Canny edge detection with different parameters
    edges = cv2.Canny(thresh, 50, 150)
    
   
    # Find contours with different retrieval mode for better results
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Look for board contour among top candidates
    board_contour = None
    for contour in contours[:10]:  # Check top 10 contours
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4:
            # Additional check: verify it's roughly square-shaped
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                # Accept if aspect ratio is close to 1 (square)
                if 0.7 <= aspect_ratio <= 1.3:
                    board_contour = approx
                    break
    
    if board_contour is None:
        raise Exception("Chessboard not found.")
    
    

    # Improved point ordering for perspective transform
    rect = order_points(board_contour)
    (tl, tr, br, bl) = rect
    
    # Calculate the width of the new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Calculate the height of the new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Ensure square output for the chessboard
    side = max(max_width, max_height)
    
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (side, side))
   
    
    return warped


def order_points(pts):
    """
    Order points in clockwise order starting from top-left.
    Important for correct perspective transform.
    """
    # Convert to the right format
    if len(pts) == 4:
        pts = pts.reshape(4, 2)
    
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def extract_squares(board_img):
    """Extract individual squares from a chessboard image"""
    # Ensure the image is square by using the minimum dimension
    min_dim = min(board_img.shape[0], board_img.shape[1])
    board_img = board_img[:min_dim, :min_dim]
    
    square_size = board_img.shape[0] // 8
    squares = []
    
    for row in range(8):
        for col in range(8):
            square = board_img[
                row * square_size:(row + 1) * square_size,
                col * square_size:(col + 1) * square_size
            ]
            squares.append(square)
    
    return squares


def ensure_directories_exist(folder_names):
    """Create directories for chess pieces if they don't exist"""
    unique_folders = set(folder_names)
    for folder in unique_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


# Main processing
if __name__ == "__main__":
    # Define folder names for the standard chess layout
    folder_names = [
        'dark_rook', 'dark_knight', 'dark_bishop', 'dark_queen', 'dark_king', 'dark_bishop', 'dark_knight', 'dark_rook',
        'dark_pawn', 'dark_pawn', 'dark_pawn', 'dark_pawn', 'dark_pawn', 'dark_pawn', 'dark_pawn', 'dark_pawn',
        'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark',
        'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light',
        'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark',
        'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light', 'empty_dark', 'empty_light',
        'light_pawn', 'light_pawn', 'light_pawn', 'light_pawn', 'light_pawn', 'light_pawn', 'light_pawn', 'light_pawn',
        'light_rook', 'light_knight', 'light_bishop', 'light_queen', 'light_king', 'light_bishop', 'light_knight', 'light_rook'
    ]
    
    # Ensure directories exist
    ensure_directories_exist(folder_names)
    
    # Process a single image
    image_path = 'cb.png'  # Change to your image path
    debug_mode = True  # Set to False to disable debug visualizations
    
    try:
        # Find and extract the chessboard
        board_img = find_chessboard(image_path, debug=debug_mode)
        
        # Extract individual squares
        squares = extract_squares(board_img)
        
        # Track what we've saved to avoid duplicates
        saved_pieces = set()
        cd, cl = (0, 0)  # Counters for dark and light pawns
        
        # Save the squares
        for idx, square in enumerate(squares):
            square_img = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
            
            folder_name = folder_names[idx]
            
            # Save logic matching your original code
            if folder_name.split('_')[0] == "empty" and folder_name not in saved_pieces:
                saved_pieces.add(folder_name)
                output_path = f'{folder_name}/square_{idx}.png'
                square_img.save(output_path)
                print(output_path)
            elif folder_name.split('_')[0] != "empty" and folder_name.split('_')[1] != "pawn":
                output_path = f'{folder_name}/square_{idx}.png'
                square_img.save(output_path)
                print(output_path)
            elif folder_name == "dark_pawn" and cd < 2:
                output_path = f'{folder_name}/square_{idx}.png'
                square_img.save(output_path)
                print(output_path)
                cd += 1
            elif folder_name == "light_pawn" and cl < 2:
                output_path = f'{folder_name}/square_{idx}.png'
                square_img.save(output_path)
                print(output_path)
                cl += 1
        
        print(f"Successfully processed {image_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        
    print(f"Failed to process {image_path}: {e}")
 