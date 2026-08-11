import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from generator.wav2vec2 import Wav2VecModel
from generator.FMT import FlowMatchingTransformer


class FMGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fps = opt.fps
        self.rank = opt.rank
        
        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_frames_for_clip + self.num_prev_frames
        
        self.audio_input_dim = 768 if opt.only_last_features else 12 * 768
        
        # Components
        self.audio_encoder = AudioEncoder(opt)
        self.fmt = FlowMatchingTransformer(opt)
        
        # Projections
        self.audio_projection = self._make_projection(self.audio_input_dim, opt.dim_c)
        self.gaze_projection = self._make_projection(2, opt.dim_c)
        self.pose_projection = self._make_projection(3, opt.dim_c)
        self.cam_projection = self._make_projection(3, opt.dim_c)

        self.odeint_kwargs = {
            'atol': opt.ode_atol,
            'rtol': opt.ode_rtol,
            'method': opt.torchdiffeq_ode_method
        }
        
        self._print_model_stats()

    def _make_projection(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def _print_model_stats(self):
        trainable = sum(p.numel() for p in self.fmt.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.fmt.parameters())
        print(f"\n[Model Stats] Parameters: {total:,} | Trainable: {trainable:,}")

    def forward(self, batch, t):
        x, prev_x = batch["m_now"], batch["m_prev"]
        a, prev_a = batch["a_now"], batch["a_prev"]
        m_ref = batch["m_ref"]
        
        gaze, prev_gaze = batch["gaze"], batch["gaze_prev"]
        pose, prev_pose = batch["pose"], batch["pose_prev"]
        cam, prev_cam = batch["cam"], batch["cam_prev"]

        bs = x.size(0)

        if not self.opt.only_last_features:
            a = a.view(bs, self.num_frames_for_clip, -1)
            prev_a = prev_a.view(bs, self.num_prev_frames, -1)

        # Projections
        a = self.audio_projection(a)
        prev_a = self.audio_projection(prev_a)
        gaze = self.gaze_projection(gaze)
        prev_gaze = self.gaze_projection(prev_gaze)
        pose = self.pose_projection(pose)
        prev_pose = self.pose_projection(prev_pose)
        cam = self.cam_projection(cam)
        prev_cam = self.cam_projection(prev_cam)

        pred = self.fmt(
            t, x.squeeze(), a, prev_x, prev_a, m_ref,
            gaze=gaze, prev_gaze=prev_gaze,
            pose=pose, prev_pose=prev_pose,
            cam=cam, prev_cam=prev_cam
        )

        return pred[:, self.num_prev_frames:, ...]

    def _align_sequence(self, tensor, target_len):
        """Helper to crop or pad sequences to target length."""
        if tensor is None:
            return None
            
        tensor = tensor.to(self.rank)
        curr_len = tensor.shape[0]
        
        if curr_len > target_len:
            return tensor[:target_len]
        elif curr_len < target_len:
            pad_len = target_len - curr_len
            padding = torch.zeros(pad_len, tensor.shape[1], device=tensor.device)
            return torch.cat([tensor, padding], dim=0)
        return tensor

    @torch.no_grad()
    def sample(self, data, a_cfg_scale=1.0, nfe=10, seed=None):
        a, ref_x = data['a'], data['ref_x']
        gaze_raw = data.get('gaze')
        pose_raw = data.get('pose')
        cam_raw = data.get('cam')
        
        B = a.shape[0]
        device = self.rank
        time_steps = torch.linspace(0, 1, nfe, device=device)

        # Process Audio
        a = a.to(device)
        T = math.ceil(a.shape[-1] * self.fps / self.opt.sampling_rate)
        a = self.audio_encoder.inference(a, seq_len=T)
        a = self.audio_projection(a)

        # Process Conditions (Gaze, Pose, Cam)
        gaze = self._align_sequence(gaze_raw, T)
        pose = self._align_sequence(pose_raw, T)
        cam = self._align_sequence(cam_raw, T)

        # Project or Create Null Embeddings
        if gaze is not None:
            gaze = self.gaze_projection(gaze).unsqueeze(0)
        else:
            gaze = torch.zeros(B, T, self.opt.dim_c, device=device)

        if pose is not None:
            pose = self.pose_projection(pose).unsqueeze(0)
        else:
            pose = torch.zeros(B, T, self.opt.dim_c, device=device)

        if cam is not None:
            cam = self.cam_projection(cam).unsqueeze(0)
        else:
            cam = torch.zeros(B, T, self.opt.dim_c, device=device)

        # Generation Loop
        sample = []
        num_chunks = int(math.ceil(T / self.num_frames_for_clip))

        for t in range(num_chunks):
            # Setup Initial Noise
            if self.opt.fix_noise_seed:
                current_seed = self.opt.seed if seed is None else seed
                g = torch.Generator(device)
                g.manual_seed(current_seed)
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=device, generator=g)
            else:
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=device)

            # Setup Previous Context
            if t == 0:
                prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_c, device=device)
                prev_a_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device)
                prev_gaze_t = torch.zeros(B, self.num_prev_frames, gaze.shape[-1], device=device)
                prev_pose_t = torch.zeros(B, self.num_prev_frames, pose.shape[-1], device=device)
                prev_cam_t = torch.zeros(B, self.num_prev_frames, cam.shape[-1], device=device)
            else:
                prev_x_t = sample_t[:, -self.num_prev_frames:]
                prev_a_t = a_t[:, -self.num_prev_frames:]
                prev_gaze_t = gaze_t[:, -self.num_prev_frames:]
                prev_pose_t = pose_t[:, -self.num_prev_frames:]
                prev_cam_t = cam_t[:, -self.num_prev_frames:]

            # Slice Current Window
            start_idx = t * self.num_frames_for_clip
            end_idx = (t + 1) * self.num_frames_for_clip
            
            a_t = a[:, start_idx:end_idx]
            gaze_t = gaze[:, start_idx:end_idx]
            pose_t = pose[:, start_idx:end_idx]
            cam_t = cam[:, start_idx:end_idx]

            # Pad last chunk if necessary
            current_chunk_len = a_t.shape[1]
            if current_chunk_len < self.num_frames_for_clip:
                pad_len = self.num_frames_for_clip - current_chunk_len
                
                def pad_tensor(tensor):
                    last = tensor[:, -1:, :].expand(-1, pad_len, -1)
                    return torch.cat([tensor, last], dim=1)

                a_t = pad_tensor(a_t)
                gaze_t = pad_tensor(gaze_t)
                pose_t = pad_tensor(pose_t)
                cam_t = pad_tensor(cam_t)

            # ODE Solver Function
            def sample_chunk(tt, zt):
                out = self.fmt.forward_with_cfg(
                    t=tt.unsqueeze(0),
                    x=zt,
                    a=a_t,
                    prev_x=prev_x_t,
                    prev_a=prev_a_t,
                    ref_x=ref_x,
                    gaze=gaze_t, prev_gaze=prev_gaze_t,
                    pose=pose_t, prev_pose=prev_pose_t,
                    cam=cam_t, prev_cam=prev_cam_t,
                    a_cfg_scale=a_cfg_scale,
                )
                return out[:, self.num_prev_frames:]

            trajectory_t = odeint(sample_chunk, x0, time_steps, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            sample.append(sample_t)

        sample = torch.cat(sample, dim=1)[:, :T]
        return sample


class AudioEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate

        self.num_frames_for_clip = int(opt.wav2vec_sec * self.fps)
        self.num_prev_frames = int(opt.num_prev_frames)

        self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        self.wav2vec2.feature_extractor._freeze_parameters()

        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def _pad_audio(self, a, target_frames):
        """Pads audio tensor to match the target frame count required by the model."""
        target_samples = int(target_frames * self.sampling_rate / self.fps)
        if a.shape[1] % target_samples != 0:
            diff = target_samples - (a.shape[1] % target_samples)
            a = F.pad(a, (0, diff), mode='replicate')
        return a

    def get_wav2vec2_feature(self, a, seq_len):
        out = self.wav2vec2(a, seq_len=seq_len, output_hidden_states=not self.only_last_features)
        
        if self.only_last_features:
            return out.last_hidden_state
        else:
            # Stack hidden states: (B, Layers, T, C) -> (B, T, Layers, C) -> (B, T, C')
            feat = torch.stack(out.hidden_states[1:], dim=1)
            feat = feat.permute(0, 2, 1, 3)
            return feat.reshape(feat.shape[0], feat.shape[1], -1)

    def forward(self, a, prev_a=None):
        total_frames = self.num_frames_for_clip
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
            total_frames += self.num_prev_frames

        a = self._pad_audio(a, total_frames)
        return self.get_wav2vec2_feature(a, seq_len=total_frames)

    @torch.no_grad()
    def inference(self, a, seq_len):
        a = self._pad_audio(a, seq_len)
        return self.get_wav2vec2_feature(a, seq_len=seq_len)