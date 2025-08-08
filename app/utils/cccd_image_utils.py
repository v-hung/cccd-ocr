from pathlib import Path
import numpy as np
import cv2
import datetime
import os
from configs import UPLOAD_DIR
from utils.image_utils import load_image
from constants.cccd_const import REGION_FIELDS_BOX

def group_edges_by_orientation(edges, img_shape):
    height, width = img_shape[:2]
    horizontal_edges = []
    vertical_edges = []

    for length, pt1, pt2 in edges:
        dx = abs(pt2[0] - pt1[0])
        dy = abs(pt2[1] - pt1[1])
        if dx >= dy:
            horizontal_edges.append((length, pt1, pt2))
        else:
            vertical_edges.append((length, pt1, pt2))

    top_edges = []
    bottom_edges = []
    for edge in horizontal_edges:
        _, pt1, pt2 = edge
        avg_y = (pt1[1] + pt2[1]) / 2
        if avg_y < height / 2:
            top_edges.append(edge)
        else:
            bottom_edges.append(edge)

    left_edges = []
    right_edges = []
    for edge in vertical_edges:
        _, pt1, pt2 = edge
        avg_x = (pt1[0] + pt2[0]) / 2
        if avg_x < width / 2:
            left_edges.append(edge)
        else:
            right_edges.append(edge)

    def pick_longest(edges_group):
        return max(edges_group, key=lambda e: e[0]) if edges_group else None

    selected_edges = []
    for group in [top_edges, bottom_edges, left_edges, right_edges]:
        edge = pick_longest(group)
        if edge:
            selected_edges.append(edge)

    return selected_edges

def compute_intersection(p1, p2, p3, p4):
    """
    Tính giao điểm giữa hai đoạn thẳng (p1, p2) và (p3, p4)
    """
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]

    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # song song

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return np.array([x, y], dtype=np.float32)

def find_largest_cccd_contour(img, save_folder: Path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(blurred, 127, 255, 0)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), angle = rect

        if width < height:
            width, height = height, width

        aspect_ratio = width / height
        if 1.4 < aspect_ratio < 1.8:
            cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
            cv2.imwrite(str(save_folder / "2_detected_contour.jpg"), img)
            return contour

    return None

def detect_cccd_corners(img, save_folder: Path, contour):
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    points = approx_polygon.reshape(-1, 2)

    def extract_edges(points):
        edges = []
        n = len(points)
        for i in range(n):
            pt1 = points[i]
            pt2 = points[(i + 1) % n]
            length = np.linalg.norm(pt1 - pt2)
            edges.append((length, pt1, pt2))
        return sorted(edges, key=lambda e: -e[0])

    edges = extract_edges(points)

    min_edge_length = min(img.shape[:2]) / 20
    edges = [e for e in edges if e[0] > min_edge_length]

    def draw_edges(img, edges, color=(0, 0, 255), thickness=2):
        output = img.copy()
        for _, pt1, pt2 in edges:
            cv2.line(output, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness)
        return output

    debug_img = draw_edges(img, edges)
    cv2.imwrite(str(save_folder / "3_all_detected_edges.jpg"), debug_img)

    selected_edges = group_edges_by_orientation(edges, img.shape)

    color_palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, (_, pt1, pt2) in enumerate(selected_edges):
        cv2.line(img, tuple(pt1), tuple(pt2), color_palette[i % len(color_palette)], 3)

    cv2.imwrite(str(save_folder / "4_selected_edges.jpg"), img)

    top = selected_edges[0][1:3]
    bottom = selected_edges[1][1:3]
    left = selected_edges[2][1:3]
    right = selected_edges[3][1:3]

    top_left = compute_intersection(*top, *left)
    top_right = compute_intersection(*top, *right)
    bottom_left = compute_intersection(*bottom, *left)
    bottom_right = compute_intersection(*bottom, *right)

    corners = [top_left, top_right, bottom_right, bottom_left]

    for pt in corners:
        cv2.circle(img, tuple(pt.astype(int)), 8, (255, 0, 255), -1)

    cv2.imwrite(str(save_folder / "5_detected_corners.jpg"), img)

    return corners

def warp_cccd_image(img, save_folder: Path, corners):
    pts_src = np.array(corners, dtype=np.float32)
    output_width, output_height = 856, 540
    pts_dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped_img = cv2.warpPerspective(img, transform_matrix, (output_width, output_height))

    output_path = str(save_folder / "6_warped_cccd.jpg")
    cv2.imwrite(output_path, warped_img)

    return output_path

def crop_regions(image_path, save_folder: Path, scale=2.5):
    image = cv2.imread(str(image_path))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cropped_map = {}

    for key, (x, y, w, h) in REGION_FIELDS_BOX.items():
        x, y, w, h = [int(v * scale) for v in (x, y, w, h)]

        # Cắt vùng
        cropped = resized[y:y+h, x:x+w]
        cropped_map[key] = cropped

        # Vẽ khung
        cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(str(save_folder / '7_regions_input.jpg'), resized)

    return cropped_map

def process_cccd_image(image_input):
    now = datetime.datetime.now()
    save_folder = UPLOAD_DIR / 'cccd' / now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_folder, exist_ok=True)

    img = load_image(image_input)
    if img is None:
        print("Không thể đọc ảnh. Kiểm tra đường dẫn.")
        return None

    cv2.imwrite(str(save_folder / "1_original_input.jpg"), img)

    cccd_contour = find_largest_cccd_contour(img.copy(), save_folder)
    if cccd_contour is None:
        print("Không tìm thấy contour phù hợp.")
        return None

    cccd_corners = detect_cccd_corners(img.copy(), save_folder, cccd_contour)
    warped_path = warp_cccd_image(img.copy(), save_folder, cccd_corners)

    cropped_map = crop_regions(warped_path, save_folder)

    return cropped_map