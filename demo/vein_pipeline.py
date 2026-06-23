# python vein_pipeline.py (runs on a generated synthetic image)
# python vein_pipeline.py img.png (runs on NIR image)
import sys
import numpy as np
import cv2
from skimage.filters import frangi
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve

# image quality and roi detection (heuristic, no training needed)
# IMPORTANT: need to tune them to own camera once we have real images
# Upgrade path: replace with a small MobileNet/EfficientNet classifier trained on good/bad images.
def assess_quality(img_gray):
    H, W = img_gray.shape
    r = {}

    r["sharpness"] = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    r["is_blurry"] = r["sharpness"] < 60.0

    r["mean_brightness"] = float(img_gray.mean())
    frac_dark = float((img_gray < 10).mean())
    frac_bright = float((img_gray > 245).mean())
    r["underexposed"] = r["mean_brightness"] < 25 or frac_dark > 0.55
    r["overexposed"] = r["mean_brightness"] > 225 or frac_bright > 0.30

    r["contrast"] = float(img_gray.std())
    r["low_contrast"] = r["contrast"] < 12.0

    _, fg = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    r["foreground_fraction"] = largest / float(H * W)
    r["arm_likely_present"] = r["foreground_fraction"] > 0.12

    reasons = []
    if r["is_blurry"]:              reasons.append("image too blurry")
    if r["underexposed"]:           reasons.append("underexposed / weak NIR signal")
    if r["overexposed"]:            reasons.append("overexposed / glare")
    if r["low_contrast"]:           reasons.append("low contrast")
    if not r["arm_likely_present"]: reasons.append("no clear arm/forearm region")
    r["passed"] = len(reasons) == 0
    r["reasons"] = reasons
    return r

# ROI = the arm region (largest Otsu foreground blob)
# Upgrade path: a trained detector that localises the antecubital fossa specifically.
def detect_roi(img_gray):
    _, fg = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return np.full_like(img_gray, 255)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    roi = np.where(labels == largest, 255, 0).astype(np.uint8)
    return roi

# vein segmentation (classical for now, later U-Net)
def _remove_small(mask, min_area=50):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

# CLAHE -> Frangi vesselness -> Otsu threshold -> morphology.
# no training data required. `veins_are_dark=True` for typical reflectance
# NIR (veins darker than skin). Returns (enhanced, vesselness[0..1], mask{0,1}).
def segment_veins_classical(img_gray, roi_mask=None, veins_are_dark=True):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    vness = frangi(blur.astype(float) / 255.0, sigmas=range(1, 8),
                   black_ridges=veins_are_dark)
    vness = (vness - vness.min()) / (vness.max() - vness.min() + 1e-8)

    v8 = (vness * 255).astype(np.uint8)
    _, mask = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = _remove_small(mask, min_area=50)
    return enhanced, vness, (mask > 0).astype(np.uint8)

# IMPORTANT: change when U-Net is ready
def segment_veins(img_gray, roi_mask=None, model=None):
    if model is not None:
        # later: U-Net inference 
        # prob = model.predict(img_gray); mask = (prob > thr)
        raise NotImplementedError("Plug your trained U-Net in here (see v1 guide).")
    _, _, mask = segment_veins_classical(img_gray, roi_mask=roi_mask)
    return mask


# vein graph extraction
def _neighbour_count(skel_bool):
    kernel = np.ones((3, 3), dtype=int)
    return convolve(skel_bool.astype(int), kernel, mode="constant") - skel_bool.astype(int)

# skeletonise the vein mask and derive the graph structure
# returns dict with skeleton, distance transform (radius), branch points, end points, and a label image of individual vein segments (skeleton split at branch points
def extract_graph(mask):
    m = mask.astype(bool)
    skel = skeletonize(m)
    dist = distance_transform_edt(m)
    nb = _neighbour_count(skel)
    branch_pts = skel & (nb >= 3)
    end_pts = skel & (nb == 1)

    # split the skeleton into segments by removing (dilated) branch points
    bp_dil = cv2.dilate(branch_pts.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    seg_skel = (skel.astype(np.uint8) & (1 - bp_dil)).astype(np.uint8)
    n_seg, seg_labels = cv2.connectedComponents(seg_skel, connectivity=8)

    return {
        "skeleton": skel,
        "distance": dist,
        "branch_points": branch_pts,
        "end_points": end_pts,
        "seg_labels": seg_labels,
        "n_segments": n_seg,
    }

# depth/ geometry: 2D for now, needs second nir camera (stereo): function is ready tho
def _segment_endpoints(xs, ys):
    pts = set(zip(xs.tolist(), ys.tolist()))
    eps = []
    for x, y in zip(xs, ys):
        c = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx or dy) and (x + dx, y + dy) in pts:
                    c += 1
        if c == 1:
            eps.append((x, y))
    return eps

# computing features that the ranking needs
# IMPORTANT: lengths/widths are in pixels, need to convert to mm once calibrated or have stereo
def compute_segment_geometry(graph, min_len=18):
    seg_labels = graph["seg_labels"]
    dist = graph["distance"]
    feats = []
    for i in range(1, graph["n_segments"]):
        ys, xs = np.where(seg_labels == i)
        if len(xs) < min_len:
            continue
        length_px = float(len(xs))
        widths = dist[ys, xs] * 2.0
        mean_width = float(np.mean(widths))
        max_width = float(np.max(widths))

        eps = _segment_endpoints(xs, ys)
        if len(eps) >= 2:
            (x0, y0), (x1, y1) = eps[0], eps[-1]
            end_dist = float(np.hypot(x1 - x0, y1 - y0))
        else:
            end_dist = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        straightness = float(min(end_dist / (length_px + 1e-6), 1.0))

        feats.append({
            "id": int(i),
            "length_px": length_px,
            "mean_width_px": mean_width,
            "max_width_px": max_width,
            "straightness": straightness,
            "centroid": (float(xs.mean()), float(ys.mean())),
            "pixels": (xs, ys),
            "endpoints": eps,
            "depth_available": False, # flips to True only with stereo
        })
    return feats

# stereo depth: only possibile with two calibrated, rectified nir cameras
# returns a disparity map with focal px and baseline mm and matric depth in mm
def estimate_depth_stereo(left_gray, right_gray, focal_px=None, baseline_mm=None,
                          num_disp=64, block=11):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    depth_mm = None
    if focal_px and baseline_mm:
        with np.errstate(divide="ignore"):
            depth_mm = np.where(disparity > 0, focal_px * baseline_mm / disparity, 0.0)
    return disparity, depth_mm

# sustainabitiy ranking
def _norm(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0))

def rank_candidates(feats, img_shape, top_k=3):
    H, W = img_shape
    scored = []
    for f in feats:
        width_s = _norm(f["mean_width_px"], 2.0, 12.0) # wider = easier
        length_s = _norm(f["length_px"], 20.0, 120.0) # longer straight = easier
        straight_s = f["straightness"] # straighter = easier
        cx, cy = f["centroid"]
        edge_dist = min(cx, W - cx, cy, H - cy) / (0.5 * min(H, W))
        edge_s = _norm(edge_dist, 0.05, 0.4)

        score = (0.35 * width_s + 0.25 * length_s +
                 0.30 * straight_s + 0.10 * edge_s)

        reasons = []
        reasons.append(f"{'wide' if width_s > 0.6 else 'moderate' if width_s > 0.3 else 'narrow'} "
                       f"(~{f['mean_width_px']:.1f}px)")
        reasons.append(f"{'long' if length_s > 0.6 else 'medium' if length_s > 0.3 else 'short'} "
                       f"(~{f['length_px']:.0f}px)")
        reasons.append(f"{'straight' if straight_s > 0.75 else 'slightly curved' if straight_s > 0.5 else 'curved'}")
        if edge_s < 0.4:
            reasons.append("near image edge")

        scored.append({"score": float(score), "feat": f, "reasons": reasons})

    scored.sort(key=lambda s: -s["score"])
    ranked = scored[:top_k]

    # best insertion point per candidate: widest pixel away from segment ends
    for c in ranked:
        xs, ys = c["feat"]["pixels"]
        dist_vals = np.hypot(xs - np.mean(xs), ys - np.mean(ys))
        central = dist_vals < np.percentile(dist_vals, 70) # drop the far ends
        if central.sum() == 0:
            central = np.ones_like(xs, dtype=bool)
        idx_pool = np.where(central)[0]
        # among central pixels, pick the widest (largest local radius)
        # approximate width by distance to nearest pool centroid is not ideal,
        # so just take the geometric middle pixel for stability
        mid = idx_pool[len(idx_pool) // 2]
        c["insertion_point"] = (int(xs[mid]), int(ys[mid]))
    return ranked


_CAND_COLORS = [(0, 215, 255), (255, 80, 200), (80, 200, 80)] # BGR: gold, pink, green


def visualize(img_gray, mask, graph, ranked):
    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # vein mask tint (green, translucent)
    tint = base.copy()
    tint[mask == 1] = (0, 180, 0)
    out = cv2.addWeighted(base, 0.6, tint, 0.4, 0)

    # branch points (small red dots)
    by, bx = np.where(graph["branch_points"])
    for x, y in zip(bx, by):
        cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1)

    # top candidates: colour their skeleton + mark insertion point
    for rank, c in enumerate(ranked):
        color = _CAND_COLORS[rank % len(_CAND_COLORS)]
        xs, ys = c["feat"]["pixels"]
        for x, y in zip(xs, ys):
            cv2.circle(out, (int(x), int(y)), 1, color, -1)
        px, py = c["insertion_point"]
        cv2.circle(out, (px, py), 9, color, 2)
        cv2.circle(out, (px, py), 2, color, -1)
        cv2.putText(out, f"#{rank+1}", (px + 11, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return out


def report_text(qc, ranked):
    lines = ["VEIN PIPELINE REPORT (Research Demo: NOT FOR CLINICAL USE)", ""]
    lines.append(f"Image quality: {'PASS' if qc['passed'] else 'FAIL'}")
    if qc["reasons"]:
        lines.append("issues: " + "; ".join(qc["reasons"]))
    lines.append(f"sharpness={qc['sharpness']:.0f}  brightness={qc['mean_brightness']:.0f}  "
                 f"contrast={qc['contrast']:.0f}  arm_area={qc['foreground_fraction']:.2f}")
    lines.append("")
    if not ranked:
        lines.append("No suitable vein segments found.")
        return "\n".join(lines)
    lines.append(f"Top {len(ranked)} candidate insertion sites (blood draw):")
    for rank, c in enumerate(ranked):
        px, py = c["insertion_point"]
        lines.append(f"  #{rank+1}  score={c['score']:.2f}  at (x={px}, y={py})  "
                     + ", ".join(c["reasons"]))
    lines.append("")
    lines.append("Depth: NOT estimated (mono demo). Add a 2nd NIR camera for stereo depth.")
    return "\n".join(lines)

def run_pipeline(img_gray, model=None):
    qc = assess_quality(img_gray)
    roi = detect_roi(img_gray)
    mask = segment_veins(img_gray, roi_mask=roi, model=model)
    graph = extract_graph(mask)
    feats = compute_segment_geometry(graph)
    ranked = rank_candidates(feats, img_gray.shape, top_k=3)
    overlay = visualize(img_gray, mask, graph, ranked)
    return {
        "qc": qc, "roi": roi, "mask": mask, "graph": graph,
        "features": feats, "ranked": ranked,
        "overlay_bgr": overlay, "report": report_text(qc, ranked),
    }

# synthetic test image
# fake nir forearm: bright elliptical arm on a dark bg with few darker, branching veins and noise
# for pipeline testing only
def make_synthetic_nir(h=420, w=560, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.normal(18, 4, (h, w))).clip(0, 255).astype(np.uint8)

    arm = np.zeros((h, w), np.uint8)
    cv2.ellipse(arm, (w // 2, h // 2), (int(w * 0.42), int(h * 0.30)),
                8, 0, 360, 255, -1)
    skin = (rng.normal(150, 6, (h, w))).clip(0, 255).astype(np.uint8)
    img = np.where(arm == 255, skin, img).astype(np.uint8)

    def curve(p0, p1, p2, val, thick):
        ts = np.linspace(0, 1, 60)
        pts = []
        for t in ts:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
            pts.append([int(x), int(y)])
        cv2.polylines(img, [np.array(pts)], False, val, thick, cv2.LINE_AA)

    curve((w*0.20, h*0.40), (w*0.45, h*0.30), (w*0.78, h*0.46), 92, 5)
    curve((w*0.30, h*0.62), (w*0.50, h*0.55), (w*0.74, h*0.60), 95, 4)
    curve((w*0.45, h*0.30), (w*0.52, h*0.48), (w*0.50, h*0.66), 96, 3)
    curve((w*0.24, h*0.50), (w*0.30, h*0.40), (w*0.20, h*0.40), 98, 3)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.where(arm == 255, img, (rng.normal(18, 4, (h, w))).clip(0, 255)).astype(np.uint8)
    return img

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read image: {sys.argv[1]}"); sys.exit(1)
        tag = "input"
    else:
        img = make_synthetic_nir()
        tag = "synthetic"
        cv2.imwrite("demo_input_synthetic.png", img)
        print("Wrote demo_input_synthetic.png (no image was supplied).")

    res = run_pipeline(img)
    cv2.imwrite(f"demo_output_{tag}.png", res["overlay_bgr"])
    print("\n" + res["report"])
    print(f"\nWrote demo_output_{tag}.png")