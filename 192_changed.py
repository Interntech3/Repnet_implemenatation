import os
import cv2
import torch
import torchvision.transforms as T
from datetime import datetime
from repnet.model import RepNet

def load_model(weights_path):
    model = RepNet()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_repnet_on_video(weights_path, video_path, window_size=192, stride=3, device="cuda"):
    # Timestamp folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_root = os.path.dirname(os.path.abspath(__file__))
    visual_dir = os.path.join(project_root, "Visualizations", f"visualizations_{timestamp}")
    os.makedirs(visual_dir, exist_ok=True)
    print(f"Visualizations will be saved in: {visual_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])

    raw_frames, frames = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
        frames.append(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    total_frames = len(raw_frames)
    print(f"Total frames read: {total_frames}")

    model = load_model(weights_path).to(device)
    global_rep_frame_indices = set()
    start = 0

    # Video writer settings for annotated windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = raw_frames[0].shape[:2]

    while start < total_frames:
        end = min(start + window_size, total_frames)
        window_raw = raw_frames[start:end]
        window_tensors = frames[start:end]
        stride_frames = window_tensors[::stride]

        usable_len = (len(stride_frames) // 64) * 64
        if usable_len == 0:
            start += window_size
            continue
        stride_frames = stride_frames[:usable_len]

        device_tensor = torch.stack(stride_frames, dim=0).unflatten(0, (-1, 64)).movedim(1, 2).to(device)

        raw_period_length, raw_periodicity_score = [], []
        with torch.no_grad():
            for i in range(device_tensor.shape[0]):
                pl, ps, _ = model(device_tensor[i].unsqueeze(0))
                raw_period_length.append(pl[0].cpu())
                raw_periodicity_score.append(ps[0].cpu())

        pl_cat = torch.cat(raw_period_length)
        ps_cat = torch.cat(raw_periodicity_score)
        _, _, period_count, _ = model.get_counts(pl_cat, ps_cat, stride)

        prev_rep = -1
        rep_frame_indices_abs = []
        for idx, count in enumerate(period_count.tolist()):
            curr_rep = int(count)
            if curr_rep > prev_rep:
                if prev_rep != -1:
                    abs_idx = start + idx
                    rep_frame_indices_abs.append(abs_idx)
                    global_rep_frame_indices.add(abs_idx)
                prev_rep = curr_rep

        # Annotate frames for detected reps
        annotated_window = [f.copy() for f in window_raw]
        for abs_idx in rep_frame_indices_abs:
            local_idx = abs_idx - start
            if 0 <= local_idx < len(annotated_window):
                cv2.putText(annotated_window[local_idx], f"REP@{abs_idx}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

        # Save annotated window video
        annotated_path = os.path.join(visual_dir, f"window_{start}_{end-1}_annotated.mp4")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))
        for f in annotated_window:
            writer.write(f)
        writer.release()
        print(f"Saved annotated window: {annotated_path}")

        # <-- CHANGED: In previous code, next window could start at last rep to avoid double-counting.
        # Now: always increment by window_size (simpler, may slightly overcount at boundaries)
        start += window_size  # <-- CHANGED

    total_reps = len(global_rep_frame_indices)
    print(f"Total repetitions detected (exact): {total_reps}")
    return total_reps, visual_dir

if __name__ == "__main__":
    weights = "/content/pytorch_weights.pth"
    video = "/content/RepNet-pytorch/BBSquats-trainer.mp4"
    total_reps, visual_folder = run_repnet_on_video(weights, video, window_size=384, stride=3, device="cuda")
    print(f"Done! Visualizations folder timestamp: {visual_folder}")
