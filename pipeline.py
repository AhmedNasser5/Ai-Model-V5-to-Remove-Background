"""
pipeline.py — Background Removal Pipeline v5.1
All model classes are defined here and imported by app.py (Streamlit GUI).

Usage:
    from pipeline import PipelineConfig, BackgroundRemovalPipeline, ...

VRAM Fix (v5.1):
  - SAM2 removed entirely.
  - BiRefNet forced to CPU via CONFIG.BIREFNET_DEVICE.
  - CUDA cache flushed after every inference call and after every image.
"""

import sys
import os
import logging
import warnings
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bgremoval")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class PipelineConfig:
    """
    Central configuration for the background removal pipeline.

    All parameters are defined here; no other module contains hard-coded values.

    VRAM Fix (v5.1):
      - SAM2 fully disabled — was loading into VRAM at weight 0.0.
      - BiRefNet forced to CPU via BIREFNET_DEVICE.
      - torch.cuda.empty_cache() called after every inference and every image.
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    INPUT_DIR: str = (
        r"C:\Ahmed Nasser\Artificial Intelligence"
        r"\Graduation Project\CNN\Data"
    )
    OUTPUT_DIR: str = (
        r"C:\Ahmed Nasser\Artificial Intelligence"
        r"\Graduation Project\CNN\Remove Background\Model 1"
    )

    # ── Ensemble weights (BiRefNet + rembg only; SAM2 removed) ─────────────
    W_BIREFNET: float = 0.60
    W_REMBG:    float = 0.40

    # ── BiRefNet ────────────────────────────────────────────────────────────
    # BIREFNET_DEVICE is separate from the main device so rembg can use CUDA
    # while BiRefNet stays on CPU, preventing CUDA out-of-memory errors.
    BIREFNET_MODEL_ID: str     = "ZhengPeng7/BiRefNet"
    BIREFNET_INPUT_SIZE: tuple = (1024, 1024)
    BIREFNET_MEAN: list        = [0.485, 0.456, 0.406]
    BIREFNET_STD: list         = [0.229, 0.224, 0.225]
    BIREFNET_DEVICE: str       = "cpu"   # change to "cuda" only with 8+ GB free VRAM

    # ── Mask refinement ─────────────────────────────────────────────────────
    MORPH_OPEN_KERNEL: int     = 3
    MORPH_CLOSE_KERNEL: int    = 7
    GF_RADIUS: int             = 10
    GF_EPS: float              = 1e-4
    BILATERAL_D: int           = 9
    BILATERAL_SIGMA_C: int     = 80
    BILATERAL_SIGMA_S: int     = 80
    MIN_COMPONENT_RATIO: float = 0.001

    # ── Subject selector ────────────────────────────────────────────────────
    # Components smaller than MIN_SUBJECT_RATIO of the image area are removed.
    # This discards tourists and people while keeping large monuments.
    MIN_SUBJECT_RATIO: float = 0.05

    # ── Empty-mask guard ────────────────────────────────────────────────────
    MIN_FOREGROUND_RATIO: float = 0.01

    # ── Output ──────────────────────────────────────────────────────────────
    COMPARISON_DPI: int = 150
    SUPPORTED_EXT: set  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # ── Hardware (auto-detected) ─────────────────────────────────────────────
    device: str          = "cpu"
    gpu_name: str        = None
    gpu_memory_gb: float = None

    def __init__(self):
        self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detect GPU availability and set self.device."""
        try:
            if torch.cuda.is_available():
                self.device        = "cuda"
                self.gpu_name      = torch.cuda.get_device_name(0)
                self.gpu_memory_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
                )
                log.info("GPU          : %s", self.gpu_name)
                log.info("GPU Memory   : %.2f GB", self.gpu_memory_gb)
                log.info("CUDA Version : %s", torch.version.cuda)
            else:
                log.info("GPU not available — running on CPU.")
        except Exception:
            pass
        log.info("Active device    : %s", self.device.upper())
        log.info("BiRefNet device  : %s (forced)", self.BIREFNET_DEVICE.upper())


# Module-level default config instance; can be replaced by the GUI
CONFIG = PipelineConfig()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class ImageDataLoader:
    """
    Recursively discovers all supported images under CONFIG.INPUT_DIR.
    Yields (Path, PIL.Image in RGBA mode) for each discovered file.
    """

    def __init__(self, input_dir: str):
        self.root         = Path(input_dir)
        self.image_paths: list[Path] = []
        self._discover()

    def _discover(self) -> None:
        """Walk the directory tree and collect unique image paths."""
        if not self.root.exists():
            log.error("Input directory not found: %s", self.root)
            return

        seen: set[Path] = set()
        for ext in CONFIG.SUPPORTED_EXT:
            for p in (
                list(self.root.rglob(f"*{ext}"))
                + list(self.root.rglob(f"*{ext.upper()}"))
            ):
                resolved = p.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    self.image_paths.append(p)

        log.info("Images discovered: %d  |  folder: %s", len(self.image_paths), self.root)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self):
        for path in self.image_paths:
            try:
                img = Image.open(path)
                yield path, img.convert("RGBA")
            except Exception as exc:
                log.warning("Skipped %s — could not open: %s", path.name, exc)


# ─────────────────────────────────────────────────────────────────────────────
# LARGE STRUCTURE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
class LargeStructureSelector:
    """
    Post-processes a binary mask to retain only large dominant structures.

    Connected foreground components whose pixel area falls below
    CONFIG.MIN_SUBJECT_RATIO * total_image_area are zeroed out.

    This ensures tourists, people, or small objects in front of a monument
    are not preserved as the primary foreground subject.
    """

    def select(self, mask: np.ndarray, img_shape: tuple) -> np.ndarray:
        """
        Remove small foreground components from mask.

        Parameters
        ----------
        mask      : uint8 H x W mask (0-255)
        img_shape : (height, width) of the source image

        Returns
        -------
        Filtered uint8 mask with small components zeroed.
        """
        h, w        = img_shape[:2]
        total_px    = h * w
        min_area_px = int(total_px * CONFIG.MIN_SUBJECT_RATIO)

        binary = (mask > 127).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        filtered = np.zeros_like(mask)
        kept     = 0
        removed  = 0

        for i in range(1, n):  # label 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                filtered[labels == i] = mask[labels == i]
                kept += 1
            else:
                removed += 1

        if removed > 0:
            log.info(
                "  LargeStructureSelector: kept %d component(s), removed %d small object(s).",
                kept, removed,
            )
        return filtered


subject_selector = LargeStructureSelector()


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION ENGINES
# ─────────────────────────────────────────────────────────────────────────────

class BiRefNetEngine:
    """
    Bilateral Reference Network for dichotomous image segmentation.

    Runs at BIREFNET_INPUT_SIZE internally, then resizes the mask back to
    the original image dimensions using bilinear interpolation.

    VRAM fix: always loaded on CONFIG.BIREFNET_DEVICE (default CPU) so the
    GPU stays free for rembg and does not run out of memory on large datasets.

    Reference: https://huggingface.co/ZhengPeng7/BiRefNet
    """

    def __init__(self):
        self.device    = CONFIG.BIREFNET_DEVICE
        self.model     = None
        self.transform = None
        self._load()

    def _load(self) -> None:
        """Download and initialize BiRefNet from Hugging Face."""
        log.info("Loading BiRefNet on %s...", self.device.upper())
        try:
            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms

            self.model = (
                AutoModelForImageSegmentation
                .from_pretrained(CONFIG.BIREFNET_MODEL_ID, trust_remote_code=True)
                .to(self.device)
                .eval()
            )
            self.transform = transforms.Compose([
                transforms.Resize(CONFIG.BIREFNET_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=CONFIG.BIREFNET_MEAN, std=CONFIG.BIREFNET_STD),
            ])
            log.info("BiRefNet loaded successfully on %s.", self.device.upper())
        except Exception as exc:
            log.error("BiRefNet failed to load: %s", exc)
            self.model = None

    def predict(self, pil_image: Image.Image) -> np.ndarray | None:
        """Return a uint8 alpha mask at the original image resolution."""
        if self.model is None:
            return None
        orig_w, orig_h = pil_image.size
        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)[-1].sigmoid().cpu()
        mask = (preds[0].squeeze().numpy() * 255).astype(np.uint8)

        # Release input tensor immediately to free memory
        del tensor, preds
        gc.collect()

        return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


class RembgEngine:
    """
    rembg with IS-Net general-use backbone.

    Provides a strong, fast baseline for wide-domain background removal.
    Runs on CUDA when available; falls back to CPU automatically.

    Reference: https://github.com/danielgatis/rembg
    """

    MODEL_NAME = "isnet-general-use"

    def __init__(self):
        self.session   = None
        self.remove_fn = None
        self._load()

    def _load(self) -> None:
        log.info("Loading rembg (IS-Net)...")
        try:
            from rembg import remove, new_session
            self.remove_fn = remove
            self.session   = new_session(self.MODEL_NAME)
            log.info("rembg loaded successfully.")
        except Exception as exc:
            log.error("rembg failed to load: %s", exc)

    def predict(self, pil_image: Image.Image) -> np.ndarray | None:
        """Extract the alpha channel of the rembg result as a uint8 mask."""
        if self.session is None:
            return None
        result = self.remove_fn(pil_image, session=self.session)

        # Flush GPU memory after inference to prevent VRAM accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        arr = np.array(result)
        return arr[:, :, 3] if result.mode == "RGBA" else np.array(result.convert("L"))


# ─────────────────────────────────────────────────────────────────────────────
# MASK REFINER
# ─────────────────────────────────────────────────────────────────────────────
class MaskRefiner:
    """
    Six-stage edge-preserving mask post-processor.
    All numeric parameters are read from CONFIG.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def refine(self, img_rgb: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
        """
        Refine a uint8 mask using img_rgb as a structural guide.

        Stages
        ------
        1. Morphological open  — remove noise specks
        2. Morphological close — fill internal holes
        3. Guided filter       — edge-preserving smoothing via luminance
        4. Soft threshold      — blend uncertain boundary zone (0.3–0.7)
        5. Bilateral filter    — final surface smoothing
        6. Component filter    — remove floating fragments below area threshold
        """
        m = raw_mask.astype(np.uint8)

        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CONFIG.MORPH_OPEN_KERNEL, CONFIG.MORPH_OPEN_KERNEL),
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CONFIG.MORPH_CLOSE_KERNEL, CONFIG.MORPH_CLOSE_KERNEL),
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

        guide   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        refined = self._guided_filter(
            guide, m.astype(np.float32) / 255.0,
            radius=CONFIG.GF_RADIUS, eps=CONFIG.GF_EPS,
        )

        binary          = (refined > 0.5).astype(np.float32)
        transition_zone = (refined > 0.3) & (refined < 0.7)
        result          = binary.copy()
        result[transition_zone] = refined[transition_zone]

        result_u8 = cv2.bilateralFilter(
            (result * 255).astype(np.uint8),
            d=CONFIG.BILATERAL_D,
            sigmaColor=CONFIG.BILATERAL_SIGMA_C,
            sigmaSpace=CONFIG.BILATERAL_SIGMA_S,
        )

        return self._remove_small_components(result_u8)

    def _guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int,
        eps: float,
    ) -> np.ndarray:
        """
        Box-filter guided image filter (He et al., 2013).
        Smooths src while preserving edges present in guide.
        """
        ksize   = (2 * radius + 1, 2 * radius + 1)
        N       = cv2.boxFilter(np.ones_like(guide), -1, ksize)
        mean_I  = cv2.boxFilter(guide,       -1, ksize) / N
        mean_p  = cv2.boxFilter(src,         -1, ksize) / N
        mean_Ip = cv2.boxFilter(guide * src, -1, ksize) / N
        mean_II = cv2.boxFilter(guide ** 2,  -1, ksize) / N

        a = (mean_Ip - mean_I * mean_p) / (mean_II - mean_I ** 2 + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, ksize) / N
        mean_b = cv2.boxFilter(b, -1, ksize) / N
        return np.clip(mean_a * guide + mean_b, 0.0, 1.0)

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove connected foreground components whose area is below
        CONFIG.MIN_COMPONENT_RATIO of the total image area.
        """
        binary = (mask > 127).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        min_px = mask.shape[0] * mask.shape[1] * CONFIG.MIN_COMPONENT_RATIO
        clean  = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_px:
                clean[labels == i] = 1
        return (clean * mask).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE MERGER
# ─────────────────────────────────────────────────────────────────────────────
class EnsembleMerger:
    """
    Weighted-average ensemble of BiRefNet and rembg masks.

    SAM2 has been removed. If one engine fails, the surviving engine
    carries 100% of the weight automatically.
    """

    WEIGHTS: dict = {
        "birefnet": CONFIG.W_BIREFNET,
        "rembg":    CONFIG.W_REMBG,
    }

    def merge(self, masks: dict[str, np.ndarray | None]) -> np.ndarray:
        """Merge available uint8 masks into a single uint8 mask."""
        available = {
            k: v for k, v in masks.items()
            if v is not None and self.WEIGHTS.get(k, 0.0) > 0.0
        }

        if not available:
            raise ValueError("All segmentation engines failed or are disabled.")

        if len(available) == 1:
            return list(available.values())[0]

        total_w      = sum(self.WEIGHTS.get(k, 0.5) for k in available)
        weighted_sum = None

        for name, mask in available.items():
            w            = self.WEIGHTS.get(name, 0.5) / total_w
            mf           = mask.astype(np.float32) / 255.0
            weighted_sum = mf * w if weighted_sum is None else weighted_sum + mf * w

        return (weighted_sum * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class BackgroundRemovalPipeline:
    """
    End-to-end background removal pipeline targeting large architectural
    structures (pyramids, temples, statues) as foreground subjects.

    VRAM fixes applied in v5.1:
      - SAM2 engine removed entirely.
      - BiRefNet runs on CPU (CONFIG.BIREFNET_DEVICE).
      - torch.cuda.empty_cache() + gc.collect() after every image to prevent
        VRAM accumulation during large dataset processing.

    Results are written to CONFIG.OUTPUT_DIR.
    """

    OUTPUT_SUBDIRS = ("results", "masks", "comparisons")

    def __init__(self, device: str = "cpu"):
        self.device     = device
        self.output_dir = Path(CONFIG.OUTPUT_DIR)
        self._create_output_dirs()

        log.info("Initializing segmentation models...")

        # BiRefNet always loads on CPU — see CONFIG.BIREFNET_DEVICE
        self.birefnet  = BiRefNetEngine()
        self.rembg_eng = RembgEngine()
        self.refiner   = MaskRefiner(device=self.device)
        self.merger    = EnsembleMerger()

        self._reset_stats()
        log.info(
            "Pipeline ready.  GPU: %s  |  BiRefNet: %s  |  Output: %s",
            self.device.upper(), CONFIG.BIREFNET_DEVICE.upper(), self.output_dir,
        )

    def _create_output_dirs(self) -> None:
        for sub in self.OUTPUT_SUBDIRS:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def _reset_stats(self) -> None:
        self.stats = {
            "total": 0, "success": 0, "failed": 0,
            "times": [], "models_per_image": [],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Single image processing
    # ─────────────────────────────────────────────────────────────────────────
    def process_image(self, image_path: Path, pil_image: Image.Image) -> dict:
        """
        Execute the full pipeline on one image.

        Returns
        -------
        dict with keys: success, path, result, mask, comparison,
                        models_used, time_seconds, reason
        """
        t0             = time.time()
        stem           = image_path.stem
        orig_w, orig_h = pil_image.size
        img_rgb        = np.array(pil_image.convert("RGB"))

        log.info("Processing: %-40s  %dx%d", image_path.name, orig_w, orig_h)

        # Step 1: Run BiRefNet and rembg independently
        masks: dict[str, np.ndarray | None] = {}
        for name, engine in [
            ("birefnet", self.birefnet),
            ("rembg",    self.rembg_eng),
        ]:
            try:
                masks[name] = engine.predict(pil_image)
                status = "done" if masks[name] is not None else "no output"
            except Exception as exc:
                masks[name] = None
                status = f"error: {exc}"
            log.info("  %-10s  %s", name, status)

        models_used = [k for k, v in masks.items() if v is not None]
        if not models_used:
            log.warning("All engines failed for %s — skipping.", image_path.name)
            return {"success": False, "path": image_path, "reason": "all_engines_failed"}

        # Step 2: Weighted ensemble merge
        merged = self.merger.merge(masks)

        # Step 3: Empty-mask guard before refinement
        foreground_ratio = np.sum(merged > 127) / merged.size
        if foreground_ratio < CONFIG.MIN_FOREGROUND_RATIO:
            log.warning(
                "  Mask nearly empty (%.2f%%) for %s — skipping.",
                foreground_ratio * 100, image_path.name,
            )
            return {"success": False, "path": image_path, "reason": "empty_mask"}

        # Step 4: Six-stage mask refinement
        refined = self.refiner.refine(img_rgb, merged)

        # Step 5: Suppress small objects — keep only large structures
        refined = subject_selector.select(refined, img_rgb.shape)

        # Step 5b: Second empty-mask guard after structure selection
        post_ratio = np.sum(refined > 127) / refined.size
        if post_ratio < CONFIG.MIN_FOREGROUND_RATIO:
            log.warning(
                "  Mask empty after structure selection (%.2f%%) for %s — skipping.",
                post_ratio * 100, image_path.name,
            )
            return {"success": False, "path": image_path, "reason": "empty_after_selection"}

        # Step 6: Apply mask to produce RGBA transparent image
        result_rgba = self._apply_mask(pil_image, refined)

        # Step 7: Save all three output files
        res_path = self.output_dir / "results"     / f"{stem}_nobg.png"
        msk_path = self.output_dir / "masks"       / f"{stem}_mask.png"
        cmp_path = self.output_dir / "comparisons" / f"{stem}_compare.jpg"

        result_rgba.save(res_path, "PNG")
        Image.fromarray(refined).save(msk_path, "PNG")
        self._save_comparison(pil_image, result_rgba, refined, cmp_path)

        elapsed = time.time() - t0
        log.info("  Saved: %-40s  %.1fs", res_path.name, elapsed)

        self.stats["times"].append(elapsed)
        self.stats["models_per_image"].append(len(models_used))

        # Step 8: Flush GPU memory after every image to prevent VRAM accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "success":      True,
            "path":         image_path,
            "result":       res_path,
            "mask":         msk_path,
            "comparison":   cmp_path,
            "models_used":  models_used,
            "time_seconds": elapsed,
        }

    def _apply_mask(self, pil_image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Merge the refined alpha mask into the RGBA image."""
        rgba       = pil_image.convert("RGBA")
        r, g, b, _ = rgba.split()
        alpha      = Image.fromarray(mask, mode="L")
        return Image.merge("RGBA", (r, g, b, alpha))

    def _save_comparison(
        self,
        original: Image.Image,
        result: Image.Image,
        mask: np.ndarray,
        save_path: Path,
    ) -> None:
        """Save a 3-panel side-by-side comparison image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bg = Image.new("RGBA", original.size, (35, 35, 35, 255))
        bg.paste(result, mask=result.split()[3])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#0a0a0a")

        panels = [
            (original.convert("RGB"),              "Original"),
            (Image.fromarray(mask).convert("RGB"), "Alpha Mask"),
            (bg.convert("RGB"),                    "Background Removed"),
        ]

        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, color="#cccccc", fontsize=12, fontweight="bold", pad=8)
            ax.axis("off")

        plt.tight_layout(pad=1.5)
        plt.savefig(
            save_path,
            dpi=CONFIG.COMPARISON_DPI,
            bbox_inches="tight",
            facecolor="#0a0a0a",
            edgecolor="none",
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────
    def _save_report(self, results: list[dict]) -> None:
        """Serialize full statistics and per-image results to JSON."""
        times  = self.stats["times"]
        report = {
            "pipeline":        "Background Removal Pipeline — Model 1",
            "version":         "5.1",
            "timestamp":       datetime.now().isoformat(),
            "device":          self.device,
            "birefnet_device": CONFIG.BIREFNET_DEVICE,
            "output_dir":      str(self.output_dir),
            "statistics": {
                "total":              self.stats["total"],
                "successful":         self.stats["success"],
                "failed":             self.stats["failed"],
                "success_rate":       f"{self.stats['success'] / max(self.stats['total'], 1) * 100:.1f}%",
                "avg_time_per_image": f"{np.mean(times):.2f}s" if times else "N/A",
                "total_time":         f"{sum(times):.1f}s"     if times else "N/A",
                "avg_models_used":    f"{np.mean(self.stats['models_per_image']):.1f}" if times else "N/A",
            },
            "results": [
                {
                    "input":        str(r["path"]),
                    "output":       str(r.get("result", "")),
                    "success":      r["success"],
                    "reason":       r.get("reason", ""),
                    "models_used":  r.get("models_used", []),
                    "time_seconds": round(r.get("time_seconds", 0.0), 3),
                }
                for r in results
            ],
        }
        out = self.output_dir / "processing_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info("Report saved: %s", out)

    def _print_summary(self) -> None:
        times = self.stats["times"]
        sep   = "=" * 58
        log.info(sep)
        log.info("PIPELINE COMPLETE")
        log.info(sep)
        log.info("Total images    : %d", self.stats["total"])
        log.info("Successful      : %d", self.stats["success"])
        log.info("Failed          : %d", self.stats["failed"])
        log.info(
            "Success rate    : %.1f%%",
            self.stats["success"] / max(self.stats["total"], 1) * 100,
        )
        if times:
            log.info("Avg time/image  : %.2fs", np.mean(times))
            log.info("Total time      : %.1fs",  sum(times))
        log.info("Output folder   : %s", self.output_dir.resolve())
        log.info(sep)
