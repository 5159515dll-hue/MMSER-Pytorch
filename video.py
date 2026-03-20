'''
Description:
Author: Dai Lu Lu
version: 1.0
Date: 2026-01-23
LastEditors: Dai Lu Lu
LastEditTime: 2026-01-23 13:55:31
'''
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import rcParams

from experiments.motion_prosody.models import FlowVideoEncoder, VideoMAEEncoder

# ================== 中文支持 ==================
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 黑体
rcParams['axes.unicode_minus'] = False
# ================================================


def _read_video_frames(path: Path) -> list[np.ndarray]:
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise FileNotFoundError(f"无法打开视频: {path}")
	frames = []
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		frames.append(frame)
	cap.release()
	if not frames:
		raise RuntimeError("视频中未读取到帧")
	return frames


def _uniform_sample(frames: list[np.ndarray], num_frames: int) -> list[np.ndarray]:
	if num_frames <= 0:
		return []
	n = len(frames)
	if n <= num_frames:
		return frames
	idx = np.linspace(0, n - 1, num_frames).round().astype(int)
	return [frames[i] for i in idx]


def _pad_or_sample(frames: list[np.ndarray], num_frames: int) -> list[np.ndarray]:
	if not frames or num_frames <= 0:
		return []
	if len(frames) >= num_frames:
		return _uniform_sample(frames, num_frames)
	pad_count = num_frames - len(frames)
	return frames + [frames[-1]] * pad_count


def _detect_face(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
	gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
	detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
	if len(faces) == 0:
		return None
	# 取面积最大的人脸
	x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
	return int(x), int(y), int(w), int(h)


def _center_crop(frame_bgr: np.ndarray, size: int) -> np.ndarray:
	h, w = frame_bgr.shape[:2]
	size = min(size, h, w)
	y1 = (h - size) // 2
	x1 = (w - size) // 2
	return frame_bgr[y1:y1 + size, x1:x1 + size]


def _plot_frame_cube(frames: list[np.ndarray], num_slices: int = 6, title: str = "原始视频帧体（压缩堆叠示意）") -> None:
	if not frames:
		return
	slices = _uniform_sample(frames, num_slices)
	fig = plt.figure(figsize=(8, 6), constrained_layout=True)
	ax = fig.add_subplot(111, projection="3d")
	for i, frame in enumerate(slices):
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (160, 100))
		h, w = img.shape[:2]
		x = np.linspace(0, w, w)
		y = np.linspace(h, 0, h)
		X, Y = np.meshgrid(x, y)
		Z = np.full_like(X, i, dtype=float)
		ax.plot_surface(
			X,
			Z,
			Y,
			rstride=1,
			cstride=1,
			facecolors=img / 255.0,
			shade=False,
		)
	ax.set_title(title)
	ax.set_xlabel("W")
	ax.set_ylabel("帧序")
	ax.set_zlabel("H")
	ax.set_yticks([])
	ax.set_xticks([])
	ax.set_zticks([])


def main():
	video_path = Path("1.mp4")
	num_keyframes = 5
	face_size = 112

	frames = _read_video_frames(video_path)
	_plot_frame_cube(frames, num_slices=6, title="原始视频帧体（压缩堆叠示意）")
	keyframes = _uniform_sample(frames, num_keyframes)

	# 1) 关键帧均匀采样（展示为拼图）
	fig1, ax1 = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
	montage = cv2.hconcat([cv2.resize(f, (200, 120)) for f in keyframes])
	ax1.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
	ax1.set_title("关键帧均匀采样")
	ax1.axis("off")

	# 2) 人脸区域裁剪（关键帧拼图）
	face_crops = []
	for f in keyframes:
		box = _detect_face(f)
		if box is not None:
			x, y, w, h = box
			crop = f[y:y + h, x:x + w]
		else:
			crop = _center_crop(f, size=min(f.shape[:2]))
		face_crops.append(crop)

	fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
	face_montage = cv2.hconcat([cv2.resize(f, (200, 120)) for f in face_crops])
	ax2.imshow(cv2.cvtColor(face_montage, cv2.COLOR_BGR2RGB))
	ax2.set_title("人脸区域裁剪（拼图）")
	ax2.axis("off")

	# 3) 尺寸归一化（关键帧拼图）
	face_norms = [cv2.resize(f, (face_size, face_size)) for f in face_crops]
	fig3, ax3 = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
	face_norm_montage = cv2.hconcat([cv2.resize(f, (200, 120)) for f in face_norms])
	ax3.imshow(cv2.cvtColor(face_norm_montage, cv2.COLOR_BGR2RGB))
	ax3.set_title(f"尺寸归一化（{face_size}x{face_size}，拼图）")
	ax3.axis("off")

	# 4) 预处理后视频帧序列（对关键帧逐帧裁剪+归一化）
	preprocessed_frames = face_norms

	# C：分支2 外观表征（RGB Clip 体）
	_plot_frame_cube(preprocessed_frames, num_slices=6, title="C：分支2 外观表征（RGB Clip 体）")
	clip_montage = cv2.hconcat([cv2.resize(f, (200, 120)) for f in preprocessed_frames])

	# C1：RGB Clip 组帧（拼图展示）
	fig4b, ax4b = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
	ax4b.imshow(cv2.cvtColor(clip_montage, cv2.COLOR_BGR2RGB))
	ax4b.set_title("C1：RGB Clip 组帧（真实帧序列）")
	ax4b.axis("off")

	# C2：VideoMAE Encoder 输出（外观表征嵌入）
	appearance_emb = None
	try:
		video_mae_size = 224
		video_mae_frames = 16
		mae_frames = _pad_or_sample(preprocessed_frames, video_mae_frames)
		rgb_clip = np.stack([
			cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (video_mae_size, video_mae_size))
			for f in mae_frames
		], axis=0)
		rgb_clip = rgb_clip.astype(np.float32) / 255.0  # (T,H,W,3)
		rgb_clip = np.transpose(rgb_clip, (0, 3, 1, 2))  # (T,3,H,W)
		rgb_tensor = torch.from_numpy(rgb_clip).unsqueeze(0)  # (1,T,3,H,W)
		video_encoder = VideoMAEEncoder(model_name="MCG-NJU/videomae-large", freeze=True)
		video_encoder.eval()
		with torch.no_grad():
			appearance_emb = video_encoder(rgb_tensor).squeeze(0).cpu().numpy()
		fig4c, ax4c = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
		im4c = ax4c.imshow(appearance_emb.reshape(1, -1), aspect='auto', cmap='magma')
		ax4c.set_yticks([])
		ax4c.set_xlabel('嵌入维度')
		ax4c.set_title('C2：VideoMAE Encoder 输出（外观表征嵌入）')
		fig4c.colorbar(im4c, ax=ax4c, label='数值')
	except Exception:
		appearance_emb = None

	# 分支1：运动表征（相邻帧光流）
	flow_mags = []
	flow_vis = []
	flow_u_seq = []
	flow_v_seq = []
	flow_first_rgb = None
	for i in range(len(preprocessed_frames) - 1):
		prev = cv2.cvtColor(preprocessed_frames[i], cv2.COLOR_BGR2GRAY)
		next_f = cv2.cvtColor(preprocessed_frames[i + 1], cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prev, next_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		flow_mags.append(mag)
		flow_u_seq.append(flow[..., 0])
		flow_v_seq.append(flow[..., 1])
		mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		flow_vis.append(mag_norm.astype(np.uint8))
		if i == 0:
			flow_u = flow[..., 0]
			flow_v = flow[..., 1]
			u_norm = cv2.normalize(flow_u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			v_norm = cv2.normalize(flow_v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			flow_tensor_vis = np.stack([u_norm, v_norm, mag_norm.astype(np.uint8)], axis=-1)
		if flow_first_rgb is None:
			ang = np.arctan2(flow[..., 1], flow[..., 0])
			hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
			hsv[..., 0] = ((ang + np.pi) * (180 / np.pi / 2)).astype(np.uint8)
			hsv[..., 1] = 255
			hsv[..., 2] = mag_norm.astype(np.uint8)
			flow_first_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	if flow_mags:
		motion_repr = np.mean(np.stack(flow_mags, axis=0), axis=0)
	else:
		motion_repr = np.zeros((face_size, face_size), dtype=np.float32)

	# B1：相邻帧计算光流（取第一对）
	if flow_first_rgb is not None:
		fig5, ax5 = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
		ax5.imshow(flow_first_rgb)
		ax5.set_title("B1：相邻帧光流（方向+幅度）")
		ax5.axis("off")

	# B2：光流张量构建（u,v,mag 三通道）
	if 'flow_tensor_vis' in locals():
		fig6, ax6 = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
		ax6.imshow(flow_tensor_vis)
		ax6.set_title("B2：光流张量（u,v,mag）")
		ax6.axis("off")

	# B3：FlowVideoEncoder 输出（运动表征嵌入）
	if flow_mags:
		flow_stack = np.stack([
			np.stack(flow_u_seq, axis=0),
			np.stack(flow_v_seq, axis=0),
			np.stack(flow_mags, axis=0),
		], axis=0)  # (3, T-1, H, W)
		flow_tensor = torch.from_numpy(flow_stack).unsqueeze(0).to(torch.float32)
		encoder = FlowVideoEncoder(out_dim=256)
		encoder.eval()
		with torch.no_grad():
			motion_emb = encoder(flow_tensor).squeeze(0).cpu().numpy()

		fig6b, ax6b = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
		im6b = ax6b.imshow(motion_emb.reshape(1, -1), aspect='auto', cmap='magma')
		ax6b.set_yticks([])
		ax6b.set_xlabel('嵌入维度')
		ax6b.set_title('B3：FlowVideoEncoder 输出（运动表征嵌入）')
		fig6b.colorbar(im6b, ax=ax6b, label='数值')

		# F1：视频分支嵌入1 -> Projection 维度对齐
		proj_dim = 256
		torch.manual_seed(42)
		proj_f1 = torch.nn.Linear(motion_emb.shape[0], proj_dim, bias=True)
		proj_f1.eval()
		with torch.no_grad():
			f1_proj = proj_f1(torch.from_numpy(motion_emb).unsqueeze(0)).squeeze(0).cpu().numpy()
		fig6c, ax6c = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
		im6c = ax6c.imshow(f1_proj.reshape(1, -1), aspect='auto', cmap='magma')
		ax6c.set_yticks([])
		ax6c.set_xlabel('投影维度')
		ax6c.set_title('F1：视频分支嵌入1 → Projection')
		fig6c.colorbar(im6c, ax=ax6c, label='数值')

	fig7, ax7 = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
	if flow_vis:
		flow_montage = cv2.hconcat([cv2.resize(f, (200, 120)) for f in flow_vis])
		ax7.imshow(flow_montage, cmap="magma")
	else:
		ax7.imshow(motion_repr, cmap="magma")
	ax7.set_title("分支1：运动表征（光流幅度序列）")
	ax7.axis("off")

	# 分支2：外观表征嵌入的可视化（RGB 均值，作为参考）
	appearance_repr = np.mean(np.stack(preprocessed_frames, axis=0), axis=0)
	fig8, ax8 = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
	ax8.imshow(cv2.cvtColor(appearance_repr.astype(np.uint8), cv2.COLOR_BGR2RGB))
	ax8.set_title("外观表征（RGB 均值，仅可视化参考）")
	ax8.axis("off")

	# F2：视频分支嵌入2 -> Projection 维度对齐
	if appearance_emb is None:
		appearance_emb = appearance_repr.mean(axis=(0, 1))  # (3,)
	proj_dim = 256
	torch.manual_seed(42)
	proj_f2 = torch.nn.Linear(appearance_emb.shape[0], proj_dim, bias=True)
	proj_f2.eval()
	with torch.no_grad():
		f2_proj = proj_f2(torch.from_numpy(appearance_emb).unsqueeze(0)).squeeze(0).cpu().numpy()
	fig9, ax9 = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
	im9 = ax9.imshow(f2_proj.reshape(1, -1), aspect='auto', cmap='magma')
	ax9.set_yticks([])
	ax9.set_xlabel('投影维度')
	ax9.set_title('F2：视频分支嵌入2 → Projection')
	fig9.colorbar(im9, ax=ax9, label='数值')

	# P1/P2 -> CAT：特征拼接
	if 'f1_proj' in locals():
		cat_feat = np.concatenate([f1_proj, f2_proj], axis=0)
		fig10, ax10 = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
		im10 = ax10.imshow(cat_feat.reshape(1, -1), aspect='auto', cmap='viridis')
		ax10.set_yticks([])
		ax10.set_xlabel('拼接后维度')
		ax10.set_title('P1/P2 → CAT：特征拼接')
		fig10.colorbar(im10, ax=ax10, label='数值')

	plt.show()


if __name__ == "__main__":
	main()
