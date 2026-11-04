"""
Reader for data exported by rtabmap-export tool.
Provides an interface similar to RTABMapDatabase for the pipeline.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .rtabmap_reader import CameraModel

class ExportedDataReader:
    """Read data from an rtabmap-export directory (rgb/, calib/, poses.txt)"""

    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        if not self.export_dir.is_dir():
            raise ValueError(f"Export directory does not exist: {self.export_dir}")

        self.poses_file = self.export_dir / "poses.txt"
        self.rgb_dir = self.export_dir / "rgb"
        self.calib_dir = self.export_dir / "calib"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_optimized_poses(self) -> Dict[int, np.ndarray]:
        """Read poses from poses.txt
        Format: #timestamp x y z qx qy qz qw id
        """
        poses = {}
        if not self.poses_file.exists():
            print(f"Warning: poses file not found at {self.poses_file}")
            return poses

        with open(self.poses_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 9:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        node_id = int(parts[8])

                        r = ScipyRotation.from_quat([qx, qy, qz, qw])
                        transform = np.eye(4)
                        transform[:3, :3] = r.as_matrix()
                        transform[:3, 3] = [x, y, z]
                        poses[node_id] = transform
                    except ValueError:
                        continue
        return poses

    def get_node_data(self, node_id: int, load_images: bool = False, load_depth: bool = False):
        """
        Returns a mock NodeData object with camera_model populated from calib/{node_id}.yaml
        We use a minimal anonymous class to mimic NodeData structure needed for export.
        """
        class MockNode:
            def __init__(self, node_id, camera_model):
                self.id = node_id
                self.camera_model = camera_model
                self.pose = np.eye(4) # Dummy, pose is fetched from get_optimized_poses

        calib_file = self.calib_dir / f"{node_id}.yaml"
        if not calib_file.exists():
            return None

        camera_model = self._parse_calib_yaml(calib_file)
        if camera_model is None:
            return None

        return MockNode(node_id=node_id, camera_model=camera_model)

    def _parse_calib_yaml(self, path: Path) -> Optional[CameraModel]:
        """Parse rtabmap-export YAML calibration file using cv2.FileStorage."""
        try:
            import cv2

            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                print(f"Warning: Cannot open calibration file: {path}")
                return None

            w = int(fs.getNode('image_width').real())
            h = int(fs.getNode('image_height').real())

            # Read camera_matrix (rows/cols/data nodes, since .mat() may fail
            # on OpenCV YAML without 'dt' field)
            cm_node = fs.getNode('camera_matrix')
            rows = int(cm_node.getNode('rows').real())
            cols = int(cm_node.getNode('cols').real())
            data_node = cm_node.getNode('data')
            data = [data_node.at(i).real() for i in range(int(data_node.size()))]
            K = np.array(data).reshape(rows, cols)

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Read local_transform
            local_transform = np.eye(4)
            lt_node = fs.getNode('local_transform')
            if not lt_node.empty():
                lt_rows = int(lt_node.getNode('rows').real())
                lt_cols = int(lt_node.getNode('cols').real())
                lt_data_node = lt_node.getNode('data')
                lt_data = [lt_data_node.at(i).real() for i in range(int(lt_data_node.size()))]
                local_transform[:lt_rows, :lt_cols] = np.array(lt_data).reshape(lt_rows, lt_cols)

            fs.release()

            return CameraModel(
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=w, height=h,
                distortion=np.zeros(5),
                local_transform=local_transform
            )
        except Exception as e:
            print(f"Failed to parse calibration {path}: {e}")
            return None

    def export_images(self, output_dir: Path, node_ids: Optional[List[int]] = None) -> Dict[int, str]:
        """
        Symlink or copy exported images to the target directory.
        Returns mapping from node_id -> filename.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_names = {}
        for img_path in self.rgb_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue

            try:
                node_id = int(img_path.stem)
            except ValueError:
                continue

            if node_ids is not None and node_id not in node_ids:
                continue

            # Copy or symlink to output_dir
            out_name = f"{node_id:06d}{img_path.suffix}"
            out_path = output_dir / out_name

            if not out_path.exists():
                # Prefer symlink to save space, fallback to copy
                try:
                    os.symlink(img_path.absolute(), out_path)
                except OSError:
                    shutil.copy2(img_path, out_path)

            image_names[node_id] = out_name

        return image_names

    def export_colmap_reference_model(
        self,
        output_dir: Path,
        node_ids: Optional[List[int]] = None,
        image_names: Optional[Dict[int, str]] = None,
    ) -> Path:
        """Export RTABMap poses and cameras as a COLMAP text-format reference model.
        Uses exact same logic as RTABMapDatabase for compatibility.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        poses = self.get_optimized_poses()
        if node_ids is not None:
            poses = {k: v for k, v in poses.items() if k in node_ids}

        camera_models: Dict[int, CameraModel] = {}
        for node_id in poses:
            node = self.get_node_data(node_id)
            if node and node.camera_model:
                camera_models[node_id] = node.camera_model

        # Deduplicate cameras
        camera_key_to_id: Dict[tuple, int] = {}
        node_to_camera_id: Dict[int, int] = {}
        cameras_list: List[Tuple[int, str, int, int, List[float]]] = []
        next_camera_id = 1

        for node_id, cam in camera_models.items():
            params = (cam.fx, cam.fy, cam.cx, cam.cy)
            key = ("PINHOLE", cam.width, cam.height, params)

            if key not in camera_key_to_id:
                camera_key_to_id[key] = next_camera_id
                cameras_list.append((
                    next_camera_id, "PINHOLE", cam.width, cam.height,
                    list(params)
                ))
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
        with open(output_dir / "images.txt", "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            image_id = 1
            for node_id in sorted(poses.keys()):
                if node_id not in node_to_camera_id:
                    continue

                T_w_bl = poses[node_id]
                cam = camera_models[node_id]
                T_bl_cam = cam.local_transform
                T_wc = T_w_bl @ T_bl_cam

                T_cw = np.linalg.inv(T_wc)

                R_cw = T_cw[:3, :3]
                t_cw = T_cw[:3, 3]

                r = ScipyRotation.from_matrix(R_cw)
                quat_xyzw = r.as_quat()
                qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

                tx, ty, tz = t_cw[0], t_cw[1], t_cw[2]
                cam_id = node_to_camera_id[node_id]

                if image_names and node_id in image_names:
                    name = image_names[node_id]
                else:
                    name = f"{node_id:06d}.jpg"

                f.write(f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                        f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} {name}\n")
                f.write("\n")
                image_id += 1

        # Write empty points3D.txt
        with open(output_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        print(f"Exported reference model: {len(cameras_list)} cameras, "
              f"{image_id - 1} images to {output_dir}")

        return output_dir
