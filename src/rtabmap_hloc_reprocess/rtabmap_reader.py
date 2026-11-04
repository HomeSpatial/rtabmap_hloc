"""
RTABMap Database Reader

Provides functionality to read RTABMap database and extract images,
camera intrinsics, and poses.
"""

import sqlite3
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraModel:
    """Camera intrinsic parameters"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    local_transform: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class NodeData:
    """Data for a single RTABMap node/frame"""

    node_id: int
    timestamp: float
    pose: Optional[np.ndarray]  # 4x4 transform matrix (map frame)
    image_rgb: Optional[np.ndarray]
    image_depth: Optional[np.ndarray]
    camera_model: Optional[CameraModel]
    keypoints: Optional[np.ndarray]  # Nx2 array of keypoint positions
    descriptors: Optional[np.ndarray]  # NxD array of descriptors
    words: Optional[Dict[int, List[int]]]  # word_id -> list of keypoint indices


def decompress_data(data: bytes) -> bytes:
    """Decompress zlib compressed data"""
    if data is None or len(data) == 0:
        return data
    try:
        return zlib.decompress(data)
    except zlib.error:
        return data


def parse_transform(data: bytes) -> Optional[np.ndarray]:
    """Parse RTABMap transform from binary data"""
    if data is None or len(data) == 0:
        return None

    # RTABMap stores 3x4 transform as 12 floats
    if len(data) == 48:  # 12 floats * 4 bytes
        values = struct.unpack("12f", data)
        transform = np.eye(4)
        transform[:3, :4] = np.array(values).reshape(3, 4)

        # Check for null transform
        if np.allclose(transform[:3, :3], 0):
            return None
        return transform

    return None


def parse_camera_model(
    data: bytes, image_width: int = 0, image_height: int = 0
) -> Optional[CameraModel]:
    """Parse camera model from binary data.

    Supports two formats:
    1. RTABMap >= 0.18 serialized CameraModel/StereoCameraModel blobs
    2. RTABMap legacy float calibration blobs
    3. Older ad-hoc double blobs and K-matrix scans as fallbacks
    """
    if data is None or len(data) < 16:
        return None

    try:
        for parser in (
            _parse_serialized_camera_model,
            _parse_legacy_float_camera_model,
            _parse_camera_model_double,
            _parse_camera_model_kmatrix,
        ):
            camera_model = parser(data, image_width, image_height)
            if camera_model is not None:
                return camera_model
        return None

    except Exception as e:
        print(f"Warning: Failed to parse camera model: {e}")
        return None


def _valid_intrinsics(fx: float, fy: float, cx: float, cy: float) -> bool:
    return fx > 10 and fy > 10 and cx > 0 and cy > 0


def _resolve_image_size(
    parsed_width: float,
    parsed_height: float,
    image_width: int = 0,
    image_height: int = 0,
) -> Tuple[int, int]:
    width = int(image_width) if image_width > 0 else int(round(parsed_width))
    height = int(image_height) if image_height > 0 else int(round(parsed_height))
    return width, height


def _local_transform_from_floats(values) -> np.ndarray:
    local_transform = np.eye(4)
    if len(values) >= 12:
        local_transform[:3, :4] = np.array(values[:12], dtype=np.float32).reshape(3, 4)
    return local_transform


def _camera_model_from_matrix(
    matrix: np.ndarray,
    width: int,
    height: int,
    distortion: Optional[np.ndarray] = None,
    local_transform: Optional[np.ndarray] = None,
) -> Optional[CameraModel]:
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    if not _valid_intrinsics(fx, fy, cx, cy):
        return None
    return CameraModel(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        distortion=distortion if distortion is not None else np.zeros(5),
        local_transform=local_transform if local_transform is not None else np.eye(4),
    )


def _parse_serialized_camera_model(
    data: bytes, image_width: int = 0, image_height: int = 0
) -> Optional[CameraModel]:
    if len(data) < 16:
        return None
    header = struct.unpack_from("4i", data, 0)
    camera_type = header[3]
    if camera_type == 0:
        camera_model, _ = _parse_serialized_mono_camera_model(
            data, 0, len(data), image_width, image_height
        )
        return camera_model
    if camera_type == 1:
        return _parse_serialized_stereo_camera_model(data, image_width, image_height)
    return None


def _parse_serialized_mono_camera_model(
    data: bytes,
    offset: int,
    data_size: int,
    image_width: int = 0,
    image_height: int = 0,
) -> Tuple[Optional[CameraModel], int]:
    header_size = 11
    header_bytes = header_size * 4
    available = min(data_size, len(data) - offset)
    if available < header_bytes:
        return None, 0

    header = struct.unpack_from(f"{header_size}i", data, offset)
    if header[3] != 0:
        return None, 0

    k_count, d_count, r_count, p_count, local_count = header[6:11]
    if (
        k_count not in (0, 9)
        or d_count < 0
        or d_count > 32
        or r_count not in (0, 9)
        or p_count not in (0, 12)
        or local_count not in (0, 12)
    ):
        return None, 0

    required = header_bytes + 8 * (k_count + d_count + r_count + p_count)
    required += 4 * local_count
    if required > available:
        return None, 0

    index = offset + header_bytes
    matrix = None
    if k_count:
        vals = struct.unpack_from("9d", data, index)
        matrix = np.array(vals, dtype=np.float64).reshape(3, 3)
        index += 8 * k_count

    distortion = np.zeros(5)
    if d_count:
        vals = struct.unpack_from(f"{d_count}d", data, index)
        distortion = np.array(vals, dtype=np.float64)
        index += 8 * d_count

    if r_count:
        index += 8 * r_count

    if p_count:
        vals = struct.unpack_from("12d", data, index)
        matrix = np.array(vals, dtype=np.float64).reshape(3, 4)
        index += 8 * p_count

    local_transform = np.eye(4)
    if local_count:
        vals = struct.unpack_from("12f", data, index)
        local_transform = _local_transform_from_floats(vals)

    if matrix is None:
        return None, 0

    width, height = _resolve_image_size(header[4], header[5], image_width, image_height)
    camera_model = _camera_model_from_matrix(
        matrix,
        width,
        height,
        distortion=distortion,
        local_transform=local_transform,
    )
    if camera_model is None:
        return None, 0
    return camera_model, required


def _parse_serialized_stereo_camera_model(
    data: bytes, image_width: int = 0, image_height: int = 0
) -> Optional[CameraModel]:
    header_size = 10
    header_bytes = header_size * 4
    if len(data) < header_bytes:
        return None

    header = struct.unpack_from(f"{header_size}i", data, 0)
    if header[3] != 1:
        return None

    r_count, t_count, e_count, f_count = header[4:8]
    left_size, right_size = header[8:10]
    if (
        r_count not in (0, 9)
        or t_count not in (0, 3)
        or e_count not in (0, 9)
        or f_count not in (0, 9)
        or left_size < 0
        or right_size < 0
    ):
        return None

    matrix_bytes = 8 * (r_count + t_count + e_count + f_count)
    required = header_bytes + matrix_bytes + left_size + right_size
    if required > len(data):
        return None

    left_offset = header_bytes + matrix_bytes
    if left_size:
        camera_model, bytes_read = _parse_serialized_mono_camera_model(
            data, left_offset, left_size, image_width, image_height
        )
        if camera_model is not None and bytes_read:
            return camera_model

    right_offset = left_offset + left_size
    if right_size:
        camera_model, bytes_read = _parse_serialized_mono_camera_model(
            data, right_offset, right_size, image_width, image_height
        )
        if camera_model is not None and bytes_read:
            return camera_model
    return None


def _parse_legacy_float_camera_model(
    data: bytes, image_width: int = 0, image_height: int = 0
) -> Optional[CameraModel]:
    if len(data) % 4 != 0:
        return None
    count = len(data) // 4
    values = struct.unpack(f"{count}f", data)

    if count == 19:
        fx, fy, cx, cy = values[:4]
        if not _valid_intrinsics(fx, fy, cx, cy):
            return None
        width, height = _resolve_image_size(values[5], values[6], image_width, image_height)
        return CameraModel(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            local_transform=_local_transform_from_floats(values[7:19]),
        )

    if count == 17:
        fx, fy, cx, cy = values[:4]
        if not _valid_intrinsics(fx, fy, cx, cy):
            return None
        width, height = _resolve_image_size(0, 0, image_width, image_height)
        return CameraModel(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            local_transform=_local_transform_from_floats(values[5:17]),
        )

    if count >= 18 and count % 18 == 0:
        fx, fy, cx, cy = values[:4]
        if not _valid_intrinsics(fx, fy, cx, cy):
            return None
        width, height = _resolve_image_size(values[4], values[5], image_width, image_height)
        return CameraModel(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            local_transform=_local_transform_from_floats(values[6:18]),
        )

    if count >= 16 and count % 16 == 0:
        fx, fy, cx, cy = values[:4]
        if not _valid_intrinsics(fx, fy, cx, cy):
            return None
        width, height = _resolve_image_size(0, 0, image_width, image_height)
        return CameraModel(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            local_transform=_local_transform_from_floats(values[4:16]),
        )

    return None


def _parse_camera_model_double(
    data: bytes, image_width: int = 0, image_height: int = 0
) -> Optional[CameraModel]:
    if len(data) < 48:
        return None

    offset = 0

    # Read name
    name_size = struct.unpack("I", data[offset : offset + 4])[0]
    offset += 4 + name_size

    if offset + 32 > len(data):
        return None

    # Try original format: 4 doubles for fx, fy, cx, cy
    fx, fy, cx, cy = struct.unpack("4d", data[offset : offset + 32])

    if not _valid_intrinsics(fx, fy, cx, cy):
        return None

    offset += 32

    # Read image size (2 ints)
    width, height = image_width, image_height
    if offset + 8 <= len(data):
        parsed_width, parsed_height = struct.unpack("2i", data[offset : offset + 8])
        width, height = _resolve_image_size(
            parsed_width, parsed_height, image_width, image_height
        )
        offset += 8

    # Read distortion (up to 5 doubles)
    distortion = np.zeros(5)
    if offset + 40 <= len(data):
        dist_values = struct.unpack("5d", data[offset : offset + 40])
        distortion = np.array(dist_values)
        offset += 40

    # Read local transform (12 floats = 3x4 matrix)
    local_transform = np.eye(4)
    if offset + 48 <= len(data):
        lt_values = struct.unpack("12f", data[offset : offset + 48])
        local_transform = _local_transform_from_floats(lt_values)

    return CameraModel(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        distortion=distortion,
        local_transform=local_transform,
    )


def _parse_camera_model_kmatrix(
    data: bytes, image_width: int = 0, image_height: int = 0
) -> Optional[CameraModel]:
    """Fallback parser: scan for 3x3 K matrix pattern in calibration blob.

    RTABMap may serialize calibration with the full K matrix as a 3x3
    cv::Mat (9 doubles, row-major). This function scans for that pattern.
    """
    width, height = image_width, image_height
    if width <= 0 or height <= 0:
        # RTABMap matrix headers are int-aligned. Byte-wise scanning can cross
        # adjacent fields and invent plausible but bogus dimensions.
        for start in range(0, min(len(data) - 7, 80), 4):
            try:
                w, h = struct.unpack("2i", data[start : start + 8])
                if 100 <= w <= 10000 and 100 <= h <= 10000:
                    width, height = w, h
                    break
            except struct.error:
                pass

    # Scan for K matrix pattern:
    # K[0,0]=fx>0, K[0,1]≈0, K[1,0]≈0, K[1,1]=fy>0, K[2,0]≈0, K[2,1]≈0, K[2,2]≈1
    for start in range(0, len(data) - 71):
        try:
            vals = struct.unpack("9d", data[start : start + 72])
            mat = np.array(vals).reshape(3, 3)
            if (
                mat[0, 0] > 100
                and mat[1, 1] > 100
                and abs(mat[2, 2] - 1.0) < 0.001
                and abs(mat[0, 1]) < 0.001
                and abs(mat[1, 0]) < 0.001
                and abs(mat[2, 0]) < 0.001
                and abs(mat[2, 1]) < 0.001
            ):
                fx = mat[0, 0]
                fy = mat[1, 1]
                cx = mat[0, 2]
                cy = mat[1, 2]

                # Try to read local_transform from end of data (12 floats = 48 bytes)
                local_transform = np.eye(4)
                if len(data) >= start + 72 + 48:
                    end_offset = len(data) - 48
                    try:
                        lt_values = struct.unpack(
                            "12f", data[end_offset : end_offset + 48]
                        )
                        lt = np.array(lt_values).reshape(3, 4)
                        # Validate: local_transform should be a valid rotation
                        if np.allclose(np.abs(lt[:3, :3]).sum(axis=0), 1.0, atol=0.1):
                            local_transform[:3, :4] = lt
                    except struct.error:
                        pass

                return CameraModel(
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    distortion=np.zeros(5),
                    local_transform=local_transform,
                )
        except struct.error:
            pass

    return None


def decompress_image(data: bytes, is_depth: bool = False) -> Optional[np.ndarray]:
    """Decompress image from RTABMap format"""
    if data is None or len(data) == 0:
        return None

    try:
        # Try to decompress
        decompressed = decompress_data(data)

        # Convert to numpy array and decode
        nparr = np.frombuffer(decompressed, np.uint8)

        if is_depth:
            # Depth images are typically 16-bit PNG
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
    except Exception as e:
        print(f"Warning: Failed to decompress image: {e}")
        return None


class RTABMapDatabase:
    """Reader for RTABMap SQLite database"""

    def __init__(self, db_path: str, *, read_only: bool = False):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.conn = (
            sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)
            if read_only
            else sqlite3.connect(str(self.db_path))
        )
        self.conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_node_ids(self) -> List[int]:
        """Get all node IDs in the database"""
        cursor = self.conn.execute("SELECT id FROM Node ORDER BY id")
        return [row[0] for row in cursor.fetchall()]

    def get_optimized_poses(self) -> Dict[int, np.ndarray]:
        """Get optimized poses from the database."""
        cursor = self.conn.execute("SELECT opt_ids, opt_poses FROM Admin")
        row = cursor.fetchone()
        if row and row[0] and row[1]:
            try:

                def _decompress_admin(blob):
                    """RTAB-Map Admin blob: zlib data + 12-byte footer (rows/cols/type)."""
                    compressed = blob[:-12]
                    return zlib.decompress(compressed)

                ids_data = _decompress_admin(row[0])
                poses_data = _decompress_admin(row[1])
                ids = np.frombuffer(ids_data, dtype=np.int32)
                poses_flat = np.frombuffer(poses_data, dtype=np.float32).reshape(-1, 12)

                poses = {}
                for i, node_id in enumerate(ids):
                    T = np.eye(4)
                    T[:3, :4] = poses_flat[i].reshape(3, 4)
                    poses[int(node_id)] = T
                return poses
            except Exception:
                pass

        poses = {}
        cursor = self.conn.execute("SELECT id, pose FROM Node ORDER BY id")
        for row in cursor.fetchall():
            pose = parse_transform(row[1])
            if pose is not None:
                poses[row[0]] = pose
        return poses

    def get_node_data(
        self,
        node_id: int,
        load_images: bool = True,
        load_depth: bool = False,
        load_features: bool = False,
    ) -> Optional[NodeData]:
        """Get data for a specific node"""

        # Get basic node info
        cursor = self.conn.execute(
            "SELECT id, time_enter, pose FROM Node WHERE id = ?", (node_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        timestamp = row[1] if row[1] else 0.0
        pose = parse_transform(row[2])

        # Get sensor data
        image_rgb = None
        image_depth = None
        camera_model = None

        cursor = self.conn.execute(
            """SELECT image, depth, calibration
               FROM Data WHERE id = ?""",
            (node_id,),
        )
        data_row = cursor.fetchone()

        if data_row:
            if load_images and data_row[0]:
                image_rgb = decompress_image(data_row[0], is_depth=False)

            if load_depth and data_row[1]:
                image_depth = decompress_image(data_row[1], is_depth=True)

            if data_row[2]:
                img_h, img_w = 0, 0
                if image_rgb is not None:
                    img_h, img_w = image_rgb.shape[:2]
                camera_model = parse_camera_model(data_row[2], img_w, img_h)

        # Get features if requested
        keypoints = None
        descriptors = None
        words = None

        if load_features:
            cursor = self.conn.execute(
                "SELECT word_id, keypoint_x, keypoint_y, descriptor FROM Word WHERE node_id = ?",
                (node_id,),
            )

            word_rows = cursor.fetchall()
            if word_rows:
                kpts = []
                descs = []
                words = {}

                for i, wrow in enumerate(word_rows):
                    word_id = wrow[0]
                    kpts.append([wrow[1], wrow[2]])

                    if wrow[3]:
                        desc = np.frombuffer(wrow[3], dtype=np.float32)
                        descs.append(desc)

                    if word_id not in words:
                        words[word_id] = []
                    words[word_id].append(i)

                keypoints = np.array(kpts, dtype=np.float32)
                if descs:
                    descriptors = np.array(descs)

        return NodeData(
            node_id=node_id,
            timestamp=timestamp,
            pose=pose,
            image_rgb=image_rgb,
            image_depth=image_depth,
            camera_model=camera_model,
            keypoints=keypoints,
            descriptors=descriptors,
            words=words,
        )

    def export_images(
        self, output_dir: Path, node_ids: Optional[List[int]] = None
    ) -> Dict[int, str]:
        """Export all images to a directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if node_ids is None:
            node_ids = self.get_node_ids()

        image_names = {}

        for node_id in node_ids:
            node = self.get_node_data(node_id, load_images=True, load_depth=False)
            if node and node.image_rgb is not None:
                filename = f"{node_id:06d}.png"
                filepath = output_dir / filename

                # Convert RGB to BGR for cv2.imwrite
                img_bgr = cv2.cvtColor(node.image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(filepath), img_bgr)

                image_names[node_id] = filename

        return image_names

    def get_links(self) -> List[Tuple[int, int, int, np.ndarray]]:
        """Get all links between nodes

        Returns:
            List of (from_id, to_id, link_type, transform) tuples
        """
        links = []
        cursor = self.conn.execute("SELECT from_id, to_id, type, transform FROM Link")

        for row in cursor.fetchall():
            transform = parse_transform(row[3])
            links.append((row[0], row[1], row[2], transform))

        return links

    def export_colmap_reference_model(
        self,
        output_dir: Path,
        node_ids: Optional[List[int]] = None,
        image_names: Optional[Dict[int, str]] = None,
    ) -> Path:
        """Export RTABMap poses and cameras as a COLMAP text-format reference model.

        This creates cameras.txt, images.txt, and points3D.txt that can be
        used as a reference model for hloc triangulation.

        RTABMap poses are camera-to-world (T_wc). COLMAP expects
        world-to-camera (T_cw), so we invert them.

        Args:
            output_dir: Directory to write the reference model to.
            node_ids: Optional subset of node IDs to export.
            image_names: Optional mapping of node_id -> image filename.
                         If not provided, uses "{node_id:06d}.png".

        Returns:
            Path to the reference model directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        poses = self.get_optimized_poses()
        if node_ids is not None:
            poses = {k: v for k, v in poses.items() if k in node_ids}

        # Collect camera models per node
        camera_models: Dict[int, CameraModel] = {}
        for node_id in poses:
            node = self.get_node_data(node_id, load_images=False, load_depth=False)
            if node and node.camera_model:
                camera_models[node_id] = node.camera_model

        # Deduplicate cameras: group nodes by identical intrinsics
        # Key: (model, width, height, params_tuple) -> camera_id
        camera_key_to_id: Dict[tuple, int] = {}
        node_to_camera_id: Dict[int, int] = {}
        cameras_list: List[Tuple[int, str, int, int, List[float]]] = []
        next_camera_id = 1

        for node_id, cam in camera_models.items():
            # Use PINHOLE model (fx, fy, cx, cy)
            params = (cam.fx, cam.fy, cam.cx, cam.cy)
            key = ("PINHOLE", cam.width, cam.height, params)

            if key not in camera_key_to_id:
                camera_key_to_id[key] = next_camera_id
                cameras_list.append(
                    (next_camera_id, "PINHOLE", cam.width, cam.height, list(params))
                )
                next_camera_id += 1

            node_to_camera_id[node_id] = camera_key_to_id[key]

        # Write cameras.txt
        with open(output_dir / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(cameras_list)}\n")
            for cam_id, model, w, h, params in cameras_list:
                params_str = " ".join(f"{p:.10f}" for p in params)
                f.write(f"{cam_id} {model} {w} {h} {params_str}\n")

        # Write images.txt
        # COLMAP text format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # Followed by an empty line (no 2D points for reference model)
        from scipy.spatial.transform import Rotation as ScipyRotation

        with open(output_dir / "images.txt", "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            image_id = 1
            for node_id in sorted(poses.keys()):
                if node_id not in node_to_camera_id:
                    continue

                T_w_bl = poses[node_id]  # world-to-baselink (T_world_baselink)

                # Apply local_transform: camera → baselink
                # T_world_camera = T_world_baselink @ T_baselink_camera
                cam = camera_models[node_id]
                T_bl_cam = cam.local_transform  # camera → baselink
                T_wc = T_w_bl @ T_bl_cam  # camera → world

                T_cw = np.linalg.inv(T_wc)  # world → camera (COLMAP convention)

                R_cw = T_cw[:3, :3]
                t_cw = T_cw[:3, 3]

                # Convert rotation matrix to quaternion (w, x, y, z)
                r = ScipyRotation.from_matrix(R_cw)
                quat_xyzw = r.as_quat()  # scipy returns (x, y, z, w)
                qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

                tx, ty, tz = t_cw[0], t_cw[1], t_cw[2]
                cam_id = node_to_camera_id[node_id]

                if image_names and node_id in image_names:
                    name = image_names[node_id]
                else:
                    name = f"{node_id:06d}.png"

                f.write(
                    f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} {name}\n"
                )
                f.write("\n")  # Empty line for 2D points
                image_id += 1

        # Write empty points3D.txt
        with open(output_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write(
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            )

        print(
            f"Exported reference model: {len(cameras_list)} cameras, "
            f"{image_id - 1} images to {output_dir}"
        )

        return output_dir
