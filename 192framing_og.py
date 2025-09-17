import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from datetime import datetime
from repnet import utils, plots
from repnet.model import RepNet

def run_repnet_on_video(
    weights_path: str,
    video_path: str,
    strides = [3],
    device="cuda",
    no_score=False
):
    """
    Run RepNet on a video using adaptive 192-frame windows:
      - take 192 frames (indices start..start+191)
      - subsample with stride (e.g. 3) -> 64 frames -> RepNet input
      - get period_count (length 192 after get_counts repeat_interleave)
      - find frame indices where rep count increments (absolute indices)
      - if last rep at window end -> next window = start + 192
        else -> next window = last_rep_index
    Saves per-window stride-subsampled visual clips and top-level visualizations.

    Returns:
        period_count_series of the last processed window (list) and
        dict with global rep frame indices and per-window logs.
    """

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = "Visualizations/" + f"visualizations_{timestamp}"
    OUT_VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, folder_name)
    os.makedirs(OUT_VISUALIZATIONS_DIR, exist_ok=True)

    # Download if necessary
    if video_path.startswith("http"):
        print(f"Downloading video: {video_path}")
        video_file = os.path.join(PROJECT_ROOT, 'videos', os.path.basename(video_path) + '.mp4')
        os.makedirs(os.path.dirname(video_file), exist_ok=True)
        if not os.path.exists(video_file):
            utils.download_file(video_path, video_file)
    else:
        video_file = video_path

    # Read and preprocess frames (store raw_frames with labels already drawn on)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])

    print(f"Using Video: {video_file}")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    raw_frames, frames = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)  # BGR image, assumed to already have frame index drawn on it
        # transform for model input (RGB -> normalized tensor)
        frames.append(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    total_frames = len(raw_frames)
    print(f"Total frames read: {total_frames}")

    # Load RepNet model once
    model = RepNet()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # Will store global results
    global_rep_frame_indices = []  # absolute frame indices (in original video) where rep increments occur
    windows_log = []  # store info per processed window
    last_period_count_series = None
    last_best_results = None

    # We'll process using the first stride in the list (your code originally allowed multiple strides,
    # but adaptive stepping logic assumes single chosen stride per run). If multiple provided, we pick the first.
    if not strides:
        raise ValueError("At least one stride must be provided (e.g. [3])")
    stride = strides[0]
    window_size = 192
    start = 0

    while start + window_size <= total_frames:
        end = start + window_size  # exclusive
        print(f"\nProcessing window: {start} -> {end-1} (size {window_size}) with stride={stride}")

        # 1) Extract window frames
        window_raw = raw_frames[start:end]       # visual frames with indices drawn
        window_tensors = frames[start:end]       # transformed tensors (C,H,W)

        # 2) Subsample by stride -> should get exactly 64 frames if window_size==192 and stride==3
        stride_raw_frames = window_raw[::stride]
        stride_frames = window_tensors[::stride]

        # Trim safety (ensure exact multiple of 64 if video abnormal)
        usable_len = (len(stride_frames) // 64) * 64
        if usable_len == 0:
            print(f"Window {start}-{end-1}: not enough subsampled frames ({len(stride_frames)}) -> skipping")
            start += window_size
            continue
        stride_raw_frames = stride_raw_frames[:usable_len]
        stride_frames = stride_frames[:usable_len]

        # Save the stride-subsampled visual clip (so you can check frame ids like 0,3,6,...)
        stride_clip_path = os.path.join(OUT_VISUALIZATIONS_DIR, f"window_{start}_{end-1}_stride{stride}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = stride_raw_frames[0].shape[:2]
        out_stride = cv2.VideoWriter(stride_clip_path, fourcc, fps, (w, h))
        for f in stride_raw_frames:
            out_stride.write(f)
        out_stride.release()
        print(f"Saved subsampled stride clip: {stride_clip_path} (frames count: {len(stride_raw_frames)})")

        # 3) Prepare tensor for RepNet: shape should be (num_chunks, C, D=64, H, W)
        # since usable_len is multiple of 64, num_chunks = usable_len // 64
        device_tensor = torch.stack(stride_frames, dim=0).unflatten(0, (-1, 64)).movedim(1, 2).to(device)

        raw_period_length, raw_periodicity_score, embeddings = [], [], []
        with torch.no_grad():
            for i in range(device_tensor.shape[0]):
                pl, ps, em = model(device_tensor[i].unsqueeze(0))
                raw_period_length.append(pl[0].cpu())        # shape: (64, num_bins)
                raw_periodicity_score.append(ps[0].cpu())    # shape: (64, 1)
                embeddings.append(em[0].cpu())               # embeddings per 64 chunk

        pl_cat = torch.cat(raw_period_length)   # shape: (64 * num_chunks, num_bins) but here num_chunks likely 1
        ps_cat = torch.cat(raw_periodicity_score)   # shape: (64 * num_chunks, 1)
        emb_cat = torch.cat(embeddings) if embeddings else torch.empty(0)

        # 4) Get counts (this will expand to original window frame length via repeat_interleave(stride))
        confidence, period_length, period_count, periodicity_score = model.get_counts(pl_cat, ps_cat, stride)

        # period_count is cumulative counts aligned to original frames within this window:
        # length should be pl_cat.shape[0] * stride == window_size (or <= window_size depending trimming)
        period_count_values = period_count.tolist()
        last_period_count_series = period_count_values
        last_best_results = {
            "stride": stride,
            "confidence": confidence,
            "period_length": period_length,
            "period_count": period_count,
            "periodicity_score": periodicity_score,
            "embeddings": emb_cat,
            "pl_cat": pl_cat,
            "ps_cat": ps_cat
        }

        # 5) Find frame indices where repetition count increases (within-window indexes)
        rep_frame_indices_local = []
        prev_rep = -1
        for idx, count in enumerate(period_count_values):
            curr_rep = int(count)
            if curr_rep > prev_rep:
                # exclude index 0 (unless it is a real rep and you want to include)
                if idx != 0:
                    rep_frame_indices_local.append(idx)
                prev_rep = curr_rep
        # Before:Window shift: always moved by fixed window size (could double-count reps at edges)
        # After:  Window shift: moves start to last detected rep if not at window end; otherwise moves by window_size

        # Map to absolute video frame indices
        rep_frame_indices_abs = [start + idx for idx in rep_frame_indices_local]
        print(f"Window {start}-{end-1}: local rep indices: {rep_frame_indices_local}")
        print(f"Window {start}-{end-1}: absolute rep indices: {rep_frame_indices_abs}")

        # Annotate and save a small visualization that overlays rep marks on the window frames
        # We'll create an annotated copy of the window (full 192 frames) showing where reps occurred
        annotated_window = [f.copy() for f in window_raw]  # BGR images
        for abs_idx in rep_frame_indices_abs:
            local_idx = abs_idx - start
            if 0 <= local_idx < len(annotated_window):
                cv2.putText(annotated_window[local_idx], f"REP@{abs_idx}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3, cv2.LINE_AA)

        annotated_path = os.path.join(OUT_VISUALIZATIONS_DIR, f"window_{start}_{end-1}_annotated.mp4")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))
        for f in annotated_window:
            writer.write(f)
        writer.release()
        print(f"Saved annotated window: {annotated_path}")

        # Store logs
        windows_log.append({
            "window_start": start,
            "window_end": end - 1,
            "rep_frame_indices_local": rep_frame_indices_local,
            "rep_frame_indices_abs": rep_frame_indices_abs,
            "confidence": float(confidence),
            "period_count_values_len": len(period_count_values),
            "stride_clip_path": stride_clip_path,
            "annotated_path": annotated_path
        })

        # Append global rep frame indices
        global_rep_frame_indices.extend(rep_frame_indices_abs)

        # 6) Decide next start
        if len(rep_frame_indices_abs) == 0:
            # no reps found -> move to next non-overlapping window
            print("No reps detected in this window -> moving to next non-overlapping window")
            start += window_size
        else:
            last_rep_abs = rep_frame_indices_abs[-1]
            # If last rep is at/near the window's end (we use >= end-1), then move forward by window_size
            if last_rep_abs >= end - 1:
                print("Last rep at window end -> moving to next non-overlapping window")
                start += window_size
            else:
                # Shift window start to the last rep frame (to build new window from there)
                print(f"Shifting next window start to last rep index: {last_rep_abs}")
                start = last_rep_abs

    # End while loop processing windows

    if last_best_results is None:
        raise RuntimeError("No windows produced valid RepNet outputs. Check video length or stride.")

    # Create top-level visualizations using the embeddings from the last_best_results (same as before)
    if last_best_results.get("embeddings") is not None and last_best_results["embeddings"].numel() > 0:
        dist = torch.cdist(last_best_results["embeddings"], last_best_results["embeddings"], p=2) ** 2
        tsm_img = plots.plot_heatmap(dist.numpy(), log_scale=True)
        pca_img = plots.plot_pca(last_best_results["embeddings"].numpy())
        cv2.imwrite(os.path.join(OUT_VISUALIZATIONS_DIR, 'tsm.png'), tsm_img)
        cv2.imwrite(os.path.join(OUT_VISUALIZATIONS_DIR, 'pca.png'), pca_img)

    # Save repetitions visualization (use annotated frames from last window for continuity)
    # If you want a single final video of the last window's rep visualization, reuse annotated_window from above
    if windows_log:
        last_annotated = windows_log[-1].get("annotated_path")
        print(f"Saved visual outputs in {OUT_VISUALIZATIONS_DIR}")

    dict_frameIdx_Reps = {
        "total_global_reps_detected": len(global_rep_frame_indices),
        "global_rep_frame_indices": global_rep_frame_indices,
        "windows_log": windows_log,
        "stride_used": stride,
    }

    print(f"dict_frameIdx_Reps, {dict_frameIdx_Reps}")

    # Return the last window's period_count series for compatibility with your original function,
    # plus the dictionary with global rep indices and window logs.
    return last_period_count_series, dict_frameIdx_Reps


if __name__ == "__main__":
    weights = r"C:\Users\sangh\Downloads\pytorch_weights.pth"
    video = r"C:\Users\sangh\Downloads\BBSquats-trainer.mp4"
    period_series, info = run_repnet_on_video(weights, video, strides=[3], device="cuda", no_score=False)
    print("Done. Last window period_count length:", len(period_series) if period_series else 0)
    print("Info summary:", info)
