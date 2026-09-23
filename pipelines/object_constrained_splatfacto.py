"""Nerfstudio methods for the object-containment Gaussian pilot.

This module is embedded verbatim in ``object_splat_batch_pipeline.ipynb``. The
notebook writes it to Colab's temporary filesystem and registers the custom methods
through ``NERFSTUDIO_METHOD_CONFIGS``; Google Drive does not need this file.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.plugins.types import MethodSpecification


@dataclass
class ObjectConstrainedSplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto plus explicit silhouette and optional 3D containment constraints."""

    _target: Type = field(default_factory=lambda: ObjectConstrainedSplatfactoModel)

    alpha_positive_loss_mult: float = 0.20
    """Penalty for failing to render opacity inside a visible object mask."""

    alpha_outside_loss_mult: float = 0.25
    """Penalty for rendering opacity outside the relaxed object silhouette."""

    alpha_dice_loss_mult: float = 0.10
    """Whole-silhouette Dice penalty used with the two directional alpha losses."""

    outside_tolerance_pixels: int = 2
    """Mask dilation allowed before outside opacity is penalized."""

    minimum_positive_mask_pixels: int = 16
    """Blank and tiny masks preserve coordinates but do not carve occluded geometry."""

    gate_densification_by_mask: bool = True
    """Prevent center gradients outside the current silhouette from causing splits/clones."""

    enable_hull_constraint: bool = False
    support_grid_path: Optional[Path] = None
    raw_seed_path: Optional[Path] = None

    hull_support_loss_mult: float = 0.05
    hull_bbox_loss_mult: float = 0.10
    hull_min_occupancy: float = 0.50
    hull_loss_sample_count: int = 8192
    hull_loss_every: int = 1

    hard_project_outside_hull: bool = False
    hull_projection_interval: int = 100
    max_sigma_voxels: float = 1.25
    extent_loss_mult: float = 0.05
    projection_log_interval: int = 500

    enable_ellipsoid_constraint: bool = False
    """Constrain a finite oriented Gaussian contour, not only its center."""

    ellipsoid_sigma_multiplier: float = 2.0
    """Finite contour radius in standard deviations; a Gaussian has infinite support."""

    ellipsoid_fibonacci_direction_count: int = 42
    """Fibonacci-sphere probes; six principal-axis probes are added automatically."""

    ellipsoid_radial_shell_count: int = 2
    """Sample the outer contour and interior shells to catch unsupported cavities."""

    ellipsoid_support_loss_mult: float = 0.10
    ellipsoid_bbox_loss_mult: float = 0.10
    ellipsoid_min_occupancy: float = 0.50
    ellipsoid_loss_sample_count: int = 1024
    ellipsoid_loss_every: int = 2

    hard_constrain_ellipsoid: bool = False
    ellipsoid_hard_chunk_size: int = 16384
    ellipsoid_shrink_search_steps: int = 7
    ellipsoid_snap_threshold: float = 0.10
    """Snap a center to its occupied voxel before an extreme (>90%) scale reduction."""


class ObjectConstrainedSplatfactoModel(SplatfactoModel):
    """Object-aware Splatfacto used by the controlled bed pilot.

    The model keeps stock masked RGB supervision, but adds a separate target for
    the rendered accumulation image. For the hull arm, a conservative visual
    support grid is loaded in raw scan coordinates and aligned to Nerfstudio's
    internal training coordinates from the corresponding sparse seed points.
    The V9 arm additionally constrains sampled points throughout each learned
    quaternion-rotated finite-sigma ellipsoid.
    """

    config: ObjectConstrainedSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()
        self._last_center_support: Optional[torch.Tensor] = None
        self._hull_loaded = False
        self._last_projected_count = 0
        self._total_projected_count = 0
        self._last_ellipsoid_shrunk_count = 0
        self._total_ellipsoid_shrunk_count = 0
        self._last_ellipsoid_snapped_count = 0
        self._total_ellipsoid_snapped_count = 0
        self._last_boundary_inside_fraction = 1.0

        if self.config.enable_ellipsoid_constraint and not self.config.enable_hull_constraint:
            raise RuntimeError("Ellipsoid containment requires the 3D support hull.")
        if self.config.enable_ellipsoid_constraint:
            if float(self.config.ellipsoid_sigma_multiplier) <= 0:
                raise RuntimeError("ellipsoid_sigma_multiplier must be positive.")
            if int(self.config.ellipsoid_radial_shell_count) < 1:
                raise RuntimeError("ellipsoid_radial_shell_count must be at least one.")
            if int(self.config.ellipsoid_loss_sample_count) < 1:
                raise RuntimeError("ellipsoid_loss_sample_count must be at least one.")

        if self.config.enable_hull_constraint:
            self.register_buffer(
                "_ellipsoid_directions",
                self._build_ellipsoid_directions(
                    int(self.config.ellipsoid_fibonacci_direction_count)
                ),
            )
            self._load_hull_constraint()

    @staticmethod
    def _build_ellipsoid_directions(fibonacci_count: int) -> torch.Tensor:
        """Return deterministic unit directions covering the complete ellipsoid surface."""

        fibonacci_count = max(int(fibonacci_count), 8)
        index = torch.arange(fibonacci_count, dtype=torch.float32)
        z = 1.0 - 2.0 * (index + 0.5) / float(fibonacci_count)
        radius = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
        golden_angle = torch.tensor(np.pi * (3.0 - np.sqrt(5.0)), dtype=torch.float32)
        theta = index * golden_angle
        fibonacci = torch.stack(
            [radius * torch.cos(theta), radius * torch.sin(theta), z], dim=-1
        )
        axes = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        return torch.cat([axes, fibonacci], dim=0)

    @staticmethod
    def _quat_wxyz_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        """Convert gsplat's WXYZ quaternions to local-to-model rotation matrices."""

        q = F.normalize(quaternions, dim=-1)
        w, x, y, z = q.unbind(dim=-1)
        return torch.stack(
            [
                1.0 - 2.0 * (y.square() + z.square()),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x.square() + z.square()),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x.square() + y.square()),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)

    @staticmethod
    def _read_xyz(path: Path) -> np.ndarray:
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or [])
        if not {"x", "y", "z"}.issubset(names):
            raise RuntimeError(f"Hull seed PLY is missing x/y/z: {path}")
        return np.column_stack(
            [
                np.asarray(vertex["x"], dtype=np.float64),
                np.asarray(vertex["y"], dtype=np.float64),
                np.asarray(vertex["z"], dtype=np.float64),
            ]
        )

    @staticmethod
    def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_h = np.column_stack([source, np.ones(len(source), dtype=np.float64)])
        coefficients, _, _, _ = np.linalg.lstsq(source_h, target, rcond=None)
        return coefficients.astype(np.float32)

    @staticmethod
    def _apply_affine_numpy(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        points_h = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
        return points_h @ coefficients

    def _load_hull_constraint(self) -> None:
        if self.config.support_grid_path is None or self.config.raw_seed_path is None:
            raise RuntimeError(
                "Hull-constrained training requires support_grid_path and raw_seed_path."
            )

        grid_path = Path(self.config.support_grid_path)
        raw_seed_path = Path(self.config.raw_seed_path)
        if not grid_path.exists():
            raise FileNotFoundError(f"Missing object support grid: {grid_path}")
        if not raw_seed_path.exists():
            raise FileNotFoundError(f"Missing raw object sparse seed PLY: {raw_seed_path}")

        raw_seed = self._read_xyz(raw_seed_path)
        model_seed = self.means.detach().cpu().numpy().astype(np.float64)
        if len(raw_seed) != len(model_seed):
            raise RuntimeError(
                "Raw/model seed count mismatch; cannot recover the dataparser transform: "
                f"{len(raw_seed)} != {len(model_seed)}"
            )
        if len(raw_seed) < 4:
            raise RuntimeError("At least four object sparse seed points are required for hull alignment.")

        raw_to_model = self._fit_affine(raw_seed, model_seed)
        model_to_raw = self._fit_affine(model_seed, raw_seed)
        fitted = self._apply_affine_numpy(raw_seed, raw_to_model)
        fit_error = np.linalg.norm(fitted - model_seed, axis=1)
        model_span = float(np.linalg.norm(np.ptp(model_seed, axis=0)))
        fit_tolerance = max(1e-5, model_span * 1e-4)
        if not np.isfinite(fit_error).all() or float(np.max(fit_error)) > fit_tolerance:
            raise RuntimeError(
                "Could not align the raw support hull to Nerfstudio coordinates. "
                f"max residual={float(np.max(fit_error)):.6g}, tolerance={fit_tolerance:.6g}"
            )

        with np.load(grid_path, allow_pickle=False) as payload:
            occupancy = np.asarray(payload["occupancy"], dtype=bool)
            nearest = np.asarray(payload["nearest_occupied_index"], dtype=np.int32)
            origin = np.asarray(payload["origin"], dtype=np.float32)
            axes = np.asarray(payload["axes"], dtype=np.float32)
            voxel_size = float(np.asarray(payload["voxel_size"]).reshape(-1)[0])

        if occupancy.ndim != 3 or not occupancy.any():
            raise RuntimeError(f"Object support grid has no occupied 3D cells: {grid_path}")
        if nearest.shape != occupancy.shape + (3,):
            raise RuntimeError(
                f"Invalid nearest-index shape in {grid_path}: {nearest.shape} versus {occupancy.shape}"
            )
        if origin.shape != (3,) or axes.shape != (3, 3) or voxel_size <= 0:
            raise RuntimeError(f"Invalid support-grid geometry metadata: {grid_path}")

        self.register_buffer("_raw_to_model", torch.from_numpy(raw_to_model))
        self.register_buffer("_model_to_raw", torch.from_numpy(model_to_raw))
        self.register_buffer("_hull_origin", torch.from_numpy(origin))
        self.register_buffer("_hull_axes", torch.from_numpy(axes))
        self.register_buffer("_hull_occupancy", torch.from_numpy(occupancy))
        self.register_buffer("_hull_nearest", torch.from_numpy(nearest.astype(np.int64)))
        self.register_buffer(
            "_hull_shape",
            torch.tensor(occupancy.shape, dtype=torch.float32),
        )
        self.register_buffer("_hull_voxel_size", torch.tensor(voxel_size, dtype=torch.float32))

        raw_probe = np.vstack(
            [origin, *[origin + axes[:, axis] * voxel_size for axis in range(3)]]
        )
        model_probe = self._apply_affine_numpy(raw_probe, raw_to_model)
        model_voxel_sizes = np.linalg.norm(model_probe[1:] - model_probe[0], axis=1)
        model_voxel_size = float(np.median(model_voxel_sizes))
        if not np.isfinite(model_voxel_size) or model_voxel_size <= 0:
            raise RuntimeError("Could not derive the support-grid voxel size in model coordinates.")
        self.register_buffer(
            "_model_voxel_size",
            torch.tensor(model_voxel_size, dtype=torch.float32),
        )

        self._hull_loaded = True
        print(
            "[object-hull] loaded",
            tuple(int(value) for value in occupancy.shape),
            "occupied=",
            int(occupancy.sum()),
            "raw_voxel=",
            voxel_size,
            "model_voxel=",
            model_voxel_size,
            "alignment_max_error=",
            float(np.max(fit_error)),
        )

    @staticmethod
    def _apply_affine_torch(points: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((len(points), 1), dtype=points.dtype, device=points.device)
        return torch.cat([points, ones], dim=-1) @ coefficients.to(points)

    def _model_points_to_raw(self, points: torch.Tensor) -> torch.Tensor:
        return self._apply_affine_torch(points, self._model_to_raw)

    def _raw_points_to_model(self, points: torch.Tensor) -> torch.Tensor:
        return self._apply_affine_torch(points, self._raw_to_model)

    def _raw_points_to_grid(self, raw_points: torch.Tensor) -> torch.Tensor:
        local = (raw_points - self._hull_origin.to(raw_points)) @ self._hull_axes.to(raw_points)
        return local / self._hull_voxel_size.to(raw_points)

    def _lookup_hull(self, grid_points: torch.Tensor):
        rounded = torch.round(grid_points).long()
        shape = self._hull_shape.long()
        inside_bounds = ((rounded >= 0) & (rounded < shape)).all(dim=-1)
        clamped = torch.minimum(torch.maximum(rounded, torch.zeros_like(rounded)), shape - 1)
        occupied = self._hull_occupancy[
            clamped[:, 0], clamped[:, 1], clamped[:, 2]
        ]
        return inside_bounds & occupied, clamped

    def _sample_hull_occupancy(self, grid_points: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(self._hull_shape.to(grid_points) - 1.0, min=1.0)
        normalized = 2.0 * grid_points / denominator - 1.0
        sample_grid = normalized.view(1, 1, 1, -1, 3)
        volume = self._hull_occupancy.float().permute(2, 1, 0)[None, None]
        sampled = F.grid_sample(
            volume,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(-1)

    def _ellipsoid_boundary_model_points(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the learned oriented ellipsoid at the configured sigma contour."""

        directions = self._ellipsoid_directions.to(means)
        shell_count = max(int(self.config.ellipsoid_radial_shell_count), 1)
        shells = torch.linspace(
            1.0 / float(shell_count),
            1.0,
            shell_count,
            dtype=means.dtype,
            device=means.device,
        )
        probe_directions = (
            directions[None, :, :] * shells[:, None, None]
        ).reshape(-1, 3)
        sigma = torch.exp(log_scales)
        local_offsets = sigma[:, None, :] * probe_directions[None, :, :]
        local_offsets = local_offsets * float(self.config.ellipsoid_sigma_multiplier)
        if scale_factors is not None:
            local_offsets = local_offsets * scale_factors.to(means).reshape(-1, 1, 1)
        rotation = self._quat_wxyz_to_rotation_matrix(quaternions)
        model_offsets = torch.einsum("nij,ndj->ndi", rotation, local_offsets)
        return means[:, None, :] + model_offsets

    def _ellipsoid_boundary_grid_points(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        boundary_model = self._ellipsoid_boundary_model_points(
            means,
            log_scales,
            quaternions,
            scale_factors=scale_factors,
        )
        flat_raw = self._model_points_to_raw(boundary_model.reshape(-1, 3))
        flat_grid = self._raw_points_to_grid(flat_raw)
        return flat_grid.reshape(boundary_model.shape[0], boundary_model.shape[1], 3)

    def _ellipsoid_inside_hull(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ):
        boundary_grid = self._ellipsoid_boundary_grid_points(
            means,
            log_scales,
            quaternions,
            scale_factors=scale_factors,
        )
        flat_inside, _ = self._lookup_hull(boundary_grid.reshape(-1, 3))
        boundary_inside = flat_inside.reshape(boundary_grid.shape[:2])
        return boundary_inside.all(dim=-1), boundary_inside.float().mean()

    def _maximum_inside_scale_fraction(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
    ) -> torch.Tensor:
        """Find the largest uniform scale whose sampled contour remains in occupied cells."""

        low = torch.zeros(len(means), dtype=means.dtype, device=means.device)
        high = torch.ones_like(low)
        for _ in range(max(int(self.config.ellipsoid_shrink_search_steps), 1)):
            midpoint = 0.5 * (low + high)
            inside, _ = self._ellipsoid_inside_hull(
                means,
                log_scales,
                quaternions,
                scale_factors=midpoint,
            )
            low = torch.where(inside, midpoint, low)
            high = torch.where(inside, high, midpoint)
        return low

    def _capture_center_support(self, allowed_mask: torch.Tensor) -> None:
        if not self.config.gate_densification_by_mask:
            self._last_center_support = None
            return
        means2d = self.info.get("means2d") if isinstance(self.info, dict) else None
        if means2d is None:
            self._last_center_support = None
            return

        xy = means2d.detach()
        if xy.ndim == 3:
            xy = xy[0]
        if xy.ndim != 2 or xy.shape[-1] != 2:
            self._last_center_support = None
            return

        height, width = allowed_mask.shape[:2]
        pixel = torch.round(xy).long()
        valid = (
            torch.isfinite(xy).all(dim=-1)
            & (pixel[:, 0] >= 0)
            & (pixel[:, 0] < width)
            & (pixel[:, 1] >= 0)
            & (pixel[:, 1] < height)
        )
        support = torch.zeros(len(pixel), dtype=torch.bool, device=pixel.device)
        if valid.any():
            support[valid] = allowed_mask[
                pixel[valid, 1], pixel[valid, 0], 0
            ] > 0.5
        self._last_center_support = support

    def _gate_densification_gradients(self) -> None:
        if self._last_center_support is None or not isinstance(self.info, dict):
            return
        means2d = self.info.get("means2d")
        if means2d is None:
            return
        for attribute in ("absgrad", "grad"):
            gradient = getattr(means2d, attribute, None)
            if gradient is None or gradient.ndim < 2:
                continue
            if gradient.shape[-2] != len(self._last_center_support):
                continue
            shape = [1] * gradient.ndim
            shape[-2] = len(self._last_center_support)
            gate = self._last_center_support.to(gradient).view(shape)
            gradient.mul_(gate)

    def _hull_loss_dict(self) -> Dict[str, torch.Tensor]:
        if not self._hull_loaded or self.num_points == 0:
            return {}
        if self.step % max(int(self.config.hull_loss_every), 1) != 0:
            return {}

        sample_count = min(int(self.config.hull_loss_sample_count), self.num_points)
        start = (int(self.step) * 104729) % self.num_points
        index = (start + torch.arange(sample_count, device=self.means.device) * 15485863) % self.num_points
        means = self.means[index]
        raw = self._model_points_to_raw(means)
        grid = self._raw_points_to_grid(raw)
        occupancy = self._sample_hull_occupancy(grid)

        support_loss = torch.relu(float(self.config.hull_min_occupancy) - occupancy).square().mean()
        upper = self._hull_shape.to(grid) - 1.0
        bbox_violation = torch.relu(-grid) + torch.relu(grid - upper)
        bbox_loss = (bbox_violation / torch.clamp(self._hull_shape.to(grid), min=1.0)).square().mean()

        sigma_cap = self._model_voxel_size.to(self.scales) * float(self.config.max_sigma_voxels)
        largest_sigma = torch.exp(self.scales[index]).amax(dim=-1)
        extent_loss = torch.relu(largest_sigma / torch.clamp(sigma_cap, min=1e-8) - 1.0).square().mean()

        losses = {
            "hull_support_loss": support_loss * float(self.config.hull_support_loss_mult),
            "hull_bbox_loss": bbox_loss * float(self.config.hull_bbox_loss_mult),
            "hull_extent_loss": extent_loss * float(self.config.extent_loss_mult),
        }

        if (
            self.config.enable_ellipsoid_constraint
            and self.step % max(int(self.config.ellipsoid_loss_every), 1) == 0
        ):
            ellipsoid_count = min(
                int(self.config.ellipsoid_loss_sample_count), self.num_points
            )
            ellipsoid_index = index[:ellipsoid_count]
            boundary_grid = self._ellipsoid_boundary_grid_points(
                self.means[ellipsoid_index],
                self.scales[ellipsoid_index],
                self.quats[ellipsoid_index],
            )
            boundary_occupancy = self._sample_hull_occupancy(
                boundary_grid.reshape(-1, 3)
            ).reshape(boundary_grid.shape[:2])
            boundary_deficit = torch.relu(
                float(self.config.ellipsoid_min_occupancy) - boundary_occupancy
            )
            mean_deficit = boundary_deficit.square().mean()
            worst_deficit = boundary_deficit.amax(dim=-1).square().mean()

            boundary_upper = self._hull_shape.to(boundary_grid) - 1.0
            boundary_bbox_violation = torch.relu(-boundary_grid) + torch.relu(
                boundary_grid - boundary_upper
            )
            boundary_bbox_loss = (
                boundary_bbox_violation
                / torch.clamp(self._hull_shape.to(boundary_grid), min=1.0)
            ).square().mean()

            losses["hull_ellipsoid_support_loss"] = (
                0.5 * (mean_deficit + worst_deficit)
                * float(self.config.ellipsoid_support_loss_mult)
            )
            losses["hull_ellipsoid_bbox_loss"] = (
                boundary_bbox_loss * float(self.config.ellipsoid_bbox_loss_mult)
            )

        return losses

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if "mask" in batch and "accumulation" in outputs:
            mask = self._downscale_if_required(batch["mask"]).to(self.device).float()
            alpha = outputs["accumulation"].float()
            if mask.ndim == 2:
                mask = mask[..., None]
            if alpha.ndim == 2:
                alpha = alpha[..., None]
            if mask.shape != alpha.shape:
                raise RuntimeError(
                    f"Object mask/accumulation shape mismatch: {tuple(mask.shape)} != {tuple(alpha.shape)}"
                )

            positive = mask > 0.5
            tolerance = max(int(self.config.outside_tolerance_pixels), 0)
            if tolerance > 0:
                kernel = 2 * tolerance + 1
                allowed = F.max_pool2d(
                    mask.permute(2, 0, 1)[None],
                    kernel_size=kernel,
                    stride=1,
                    padding=tolerance,
                )[0].permute(1, 2, 0)
            else:
                allowed = mask
            self._capture_center_support(allowed)

            if int(positive.sum().item()) >= int(self.config.minimum_positive_mask_pixels):
                alpha_safe = alpha.clamp(1e-5, 1.0 - 1e-5)
                negative = allowed < 0.5
                positive_loss = -torch.log(alpha_safe[positive]).mean()
                if negative.any():
                    outside_loss = -torch.log1p(-alpha_safe[negative]).mean()
                else:
                    outside_loss = alpha_safe.new_zeros(())
                intersection = (alpha * mask).sum()
                dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (
                    alpha.sum() + mask.sum() + 1e-5
                )
                loss_dict["alpha_positive_loss"] = (
                    positive_loss * float(self.config.alpha_positive_loss_mult)
                )
                loss_dict["alpha_outside_loss"] = (
                    outside_loss * float(self.config.alpha_outside_loss_mult)
                )
                loss_dict["alpha_dice_loss"] = dice_loss * float(self.config.alpha_dice_loss_mult)
        else:
            self._last_center_support = None

        loss_dict.update(self._hull_loss_dict())
        return loss_dict

    @torch.no_grad()
    def _project_outside_hull(self) -> int:
        """Project escaped centers into occupied support while preserving sub-voxel position."""

        if not self._hull_loaded or self.num_points == 0:
            return 0
        raw = self._model_points_to_raw(self.means.data)
        grid = self._raw_points_to_grid(raw)
        inside, clamped = self._lookup_hull(grid)
        outside = ~inside

        if outside.any():
            nearest = self._hull_nearest[
                clamped[outside, 0], clamped[outside, 1], clamped[outside, 2]
            ].to(raw)
            rounded_local = torch.round(grid[outside])
            residual = torch.clamp(grid[outside] - rounded_local, -0.40, 0.40)
            projected_local = (nearest + residual) * self._hull_voxel_size.to(raw)
            projected_raw = self._hull_origin.to(raw) + projected_local @ self._hull_axes.to(raw).T
            self.means.data[outside] = self._raw_points_to_model(projected_raw)
        return int(outside.sum().item())

    @torch.no_grad()
    def _snap_centers_to_occupied_cell(self, indices: torch.Tensor) -> None:
        """Move severe boundary cases to the center of their current occupied cell."""

        if not len(indices):
            return
        raw = self._model_points_to_raw(self.means.data[indices])
        grid = self._raw_points_to_grid(raw)
        inside, clamped = self._lookup_hull(grid)
        if not inside.all():
            raise RuntimeError("Ellipsoid correction received a center outside the support hull.")
        snapped_local = clamped.to(raw) * self._hull_voxel_size.to(raw)
        snapped_raw = self._hull_origin.to(raw) + snapped_local @ self._hull_axes.to(raw).T
        self.means.data[indices] = self._raw_points_to_model(snapped_raw)

    @torch.no_grad()
    def _constrain_ellipsoid_extents(self) -> Dict[str, float]:
        """Shrink only Gaussians whose sampled oriented contour crosses the support volume."""

        if (
            not self._hull_loaded
            or not self.config.enable_ellipsoid_constraint
            or not self.config.hard_constrain_ellipsoid
            or self.num_points == 0
        ):
            return {"shrunk": 0, "snapped": 0, "boundary_inside_fraction": 1.0}

        shrunk_total = 0
        snapped_total = 0
        inside_sample_total = 0.0
        boundary_sample_total = 0
        chunk_size = max(int(self.config.ellipsoid_hard_chunk_size), 1)

        for start in range(0, self.num_points, chunk_size):
            end = min(start + chunk_size, self.num_points)
            chunk_index = torch.arange(start, end, device=self.means.device)
            means = self.means.data[chunk_index]
            scales = self.scales.data[chunk_index]
            quats = self.quats.data[chunk_index]
            inside, _ = self._ellipsoid_inside_hull(means, scales, quats)
            violating_local = torch.nonzero(~inside, as_tuple=False).reshape(-1)
            if len(violating_local):
                violating_index = chunk_index[violating_local]
                factors = self._maximum_inside_scale_fraction(
                    self.means.data[violating_index],
                    self.scales.data[violating_index],
                    self.quats.data[violating_index],
                )
                severe = factors < float(self.config.ellipsoid_snap_threshold)
                snapped_mask = severe.clone()
                if severe.any():
                    severe_index = violating_index[severe]
                    self._snap_centers_to_occupied_cell(severe_index)
                    factors[severe] = self._maximum_inside_scale_fraction(
                        self.means.data[severe_index],
                        self.scales.data[severe_index],
                        self.quats.data[severe_index],
                    )
                    snapped_total += int(severe.sum().item())

                # Stay just inside the last passing binary-search contour.
                factors = torch.clamp(factors * 0.98, min=1e-4, max=1.0)
                for _ in range(8):
                    corrected_inside, _ = self._ellipsoid_inside_hull(
                        self.means.data[violating_index],
                        self.scales.data[violating_index],
                        self.quats.data[violating_index],
                        scale_factors=factors,
                    )
                    if corrected_inside.all():
                        break
                    factors[~corrected_inside] = torch.clamp(
                        factors[~corrected_inside] * 0.5,
                        min=1e-4,
                    )
                corrected_inside, _ = self._ellipsoid_inside_hull(
                    self.means.data[violating_index],
                    self.scales.data[violating_index],
                    self.quats.data[violating_index],
                    scale_factors=factors,
                )
                if not corrected_inside.all():
                    newly_snapped = ~corrected_inside & ~snapped_mask
                    if newly_snapped.any():
                        self._snap_centers_to_occupied_cell(violating_index[newly_snapped])
                        snapped_total += int(newly_snapped.sum().item())
                    factors[~corrected_inside] = 1e-4
                self.scales.data[violating_index] += torch.log(factors)[:, None]
                shrunk_total += int(len(violating_index))

            _, boundary_fraction = self._ellipsoid_inside_hull(
                self.means.data[chunk_index],
                self.scales.data[chunk_index],
                self.quats.data[chunk_index],
            )
            probe_count = int(self._ellipsoid_directions.shape[0]) * max(
                int(self.config.ellipsoid_radial_shell_count), 1
            )
            inside_sample_total += float(boundary_fraction.item()) * len(chunk_index) * probe_count
            boundary_sample_total += len(chunk_index) * probe_count

        boundary_inside_fraction = (
            inside_sample_total / float(boundary_sample_total)
            if boundary_sample_total
            else 1.0
        )
        return {
            "shrunk": shrunk_total,
            "snapped": snapped_total,
            "boundary_inside_fraction": boundary_inside_fraction,
        }

    @torch.no_grad()
    def _enforce_hull_containment(self) -> Dict[str, float]:
        center_projected = self._project_outside_hull()
        sigma_cap = self._model_voxel_size.to(self.scales) * float(self.config.max_sigma_voxels)
        self.scales.data.clamp_(max=torch.log(torch.clamp(sigma_cap, min=1e-8)))
        extent_stats = self._constrain_ellipsoid_extents()
        return {
            "center_projected": center_projected,
            **extent_stats,
        }

    def step_post_backward(self, step):
        self._gate_densification_gradients()
        super().step_post_backward(step)

        containment_stats = {
            "center_projected": 0,
            "shrunk": 0,
            "snapped": 0,
            "boundary_inside_fraction": self._last_boundary_inside_fraction,
        }
        if (
            self._hull_loaded
            and self.config.hard_project_outside_hull
            and step % max(int(self.config.hull_projection_interval), 1) == 0
        ):
            containment_stats = self._enforce_hull_containment()
        self._last_projected_count = int(containment_stats["center_projected"])
        self._last_ellipsoid_shrunk_count = int(containment_stats["shrunk"])
        self._last_ellipsoid_snapped_count = int(containment_stats["snapped"])
        self._last_boundary_inside_fraction = float(
            containment_stats["boundary_inside_fraction"]
        )
        self._total_projected_count += self._last_projected_count
        self._total_ellipsoid_shrunk_count += self._last_ellipsoid_shrunk_count
        self._total_ellipsoid_snapped_count += self._last_ellipsoid_snapped_count

        if self._hull_loaded and step % max(int(self.config.projection_log_interval), 1) == 0:
            print(
                f"[object-hull] step={step} center_projected={self._last_projected_count} "
                f"total_center_projected={self._total_projected_count} "
                f"ellipsoid_shrunk={self._last_ellipsoid_shrunk_count} "
                f"total_ellipsoid_shrunk={self._total_ellipsoid_shrunk_count} "
                f"ellipsoid_snapped={self._last_ellipsoid_snapped_count} "
                f"boundary_inside_fraction={self._last_boundary_inside_fraction:.6f} "
                f"gaussians={self.num_points}"
            )

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        if self._hull_loaded and self.num_points:
            with torch.no_grad():
                sample_count = min(8192, self.num_points)
                raw = self._model_points_to_raw(self.means[:sample_count])
                inside, _ = self._lookup_hull(self._raw_points_to_grid(raw))
                metrics["hull_inside_fraction"] = inside.float().mean()
                metrics["hull_center_inside_fraction"] = inside.float().mean()
                metrics["hull_projected_last"] = torch.tensor(
                    float(self._last_projected_count), device=self.device
                )
                if self.config.enable_ellipsoid_constraint:
                    ellipsoid_count = min(1024, self.num_points)
                    ellipsoid_inside, boundary_fraction = self._ellipsoid_inside_hull(
                        self.means[:ellipsoid_count],
                        self.scales[:ellipsoid_count],
                        self.quats[:ellipsoid_count],
                    )
                    metrics["hull_ellipsoid_inside_fraction"] = (
                        ellipsoid_inside.float().mean()
                    )
                    metrics["hull_boundary_sample_inside_fraction"] = boundary_fraction
                    metrics["hull_ellipsoid_shrunk_last"] = torch.tensor(
                        float(self._last_ellipsoid_shrunk_count), device=self.device
                    )
                    metrics["hull_ellipsoid_snapped_last"] = torch.tensor(
                        float(self._last_ellipsoid_snapped_count), device=self.device
                    )
        return metrics


def _build_method_spec(
    method_name: str,
    enable_hull: bool,
    enable_ellipsoid: bool = False,
) -> MethodSpecification:
    trainer = copy.deepcopy(method_configs["splatfacto"])
    trainer.method_name = method_name
    trainer.pipeline.model = ObjectConstrainedSplatfactoModelConfig(
        enable_hull_constraint=enable_hull,
        hard_project_outside_hull=enable_hull,
        enable_ellipsoid_constraint=enable_ellipsoid,
        hard_constrain_ellipsoid=enable_ellipsoid,
        hull_support_loss_mult=0.05 if enable_hull else 0.0,
        hull_bbox_loss_mult=0.10 if enable_hull else 0.0,
        extent_loss_mult=0.05 if enable_hull else 0.0,
        ellipsoid_support_loss_mult=0.10 if enable_ellipsoid else 0.0,
        ellipsoid_bbox_loss_mult=0.10 if enable_ellipsoid else 0.0,
        use_scale_regularization=True,
        max_gauss_ratio=5.0,
        background_color="random",
    )
    if enable_ellipsoid:
        suffix = "and an oriented finite-sigma ellipsoid constrained to a multi-view support hull."
    elif enable_hull:
        suffix = "and a center-only multi-view support hull."
    else:
        suffix = "and no 3D hull constraint."
    description = (
        "Splatfacto with explicit object-opacity supervision, silhouette-gated densification, "
        + suffix
    )
    return MethodSpecification(config=trainer, description=description)


ObjectAlphaSplatfacto = _build_method_spec("object-alpha-splatfacto", enable_hull=False)
ObjectHullSplatfacto = _build_method_spec("object-hull-splatfacto", enable_hull=True)
ObjectEllipsoidSplatfacto = _build_method_spec(
    "object-ellipsoid-splatfacto",
    enable_hull=True,
    enable_ellipsoid=True,
)
