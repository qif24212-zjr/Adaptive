
Minimal Adaptive Selector — the paper-core sequential frame selection policy.

    frame_features [B, M, D]  (M candidate frames per video)
        -> Global Temporal Memory:  m_j = f_m(x_j) + temporal_position_j
                                     z_0 = W0 * mean_j(m_j) + b0
                                     h_0 = LayerNorm(z_0)
        -> Sequential Policy (per step t):
             candidate-specific conditioning (content-dependent fix):
               xbar_t = mean(selected frame features)
               delta_j = x_j - xbar_t,  interaction_j = x_j * xbar_t
               candidate_input_j = [x_j, xbar_t, delta_j, interaction_j, pos_j]
               z_candidate_j = candidate_mlp(candidate_input_j)
               logit_j = policy([h_t || z_candidate_j])   per candidate j
               logit_STOP = stop_head(h_t)                separate STOP logit
             action mask: {j <= last_selected_idx}  (monotonic temporal order;
                          subsumes "already selected" because selections are
                          strictly increasing), invalid candidates, and the
                          adaptive-STOP rules below
        -> on SELECT j:
             x_new = x_j, bar_x = mean(selected features)
             incremental = MLP([x_new || bar_x || (x_new - bar_x) || (x_new * bar_x)])
             h_{t+1}, c_{t+1} = LSTM(incremental + temporal_context_t, h_t, c_t)
        -> on STOP: episode ends
        -> adaptive stop:
             STOP forbidden at t = 0 and while count < min_selected_frames;
             episode force-ends at max_selected_frames.

This module is deliberately standalone: no CLIP, no CaptionHead, no reward,
no policy-gradient / REINFORCE, no training loop. Those come in later phases.
Integration with frame_cocap (next phase) only consumes
selected_indices / selected_mask.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["AdaptiveSelector"]


class AdaptiveSelector(nn.Module):
    def __init__(
            self,
            feature_dim: int = 512,
            hidden_size: int = 256,
            min_selected_frames: int = 2,
            max_selected_frames: int = 16,
            max_candidates: int = 1024,
            sample: bool = True,
    ):
        super().__init__()
        assert min_selected_frames >= 1, "at least one frame must be selected"
        assert max_selected_frames >= min_selected_frames
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.min_selected_frames = min_selected_frames
        self.max_selected_frames = max_selected_frames
        self.max_candidates = max_candidates
        self.sample = sample 

        self.memory_proj = nn.Linear(feature_dim, hidden_size)    
        self.memory_pos = nn.Embedding(max_candidates, hidden_size)  
        self.h0_proj = nn.Linear(hidden_size, hidden_size)          
        self.h0_ln = nn.LayerNorm(hidden_size)                      

 
        self.candidate_mlp = nn.Sequential(
            nn.Linear(4 * feature_dim + hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.policy = nn.Linear(2 * hidden_size, 1)     
        self.stop_head = nn.Linear(hidden_size, 1)      


        self.incremental_mlp = nn.Sequential(
            nn.Linear(4 * feature_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.step_pos = nn.Embedding(max_selected_frames + 1, hidden_size)  # temporal_context_t


        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)

    def forward(
            self,
            frame_features: Tensor,
            frame_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            frame_features: (B, M, D) candidate frame features.
            frame_mask:     (B, M) bool, 1 = valid candidate (default: all valid).

        Returns:
            selected_indices: (B, max_selected_frames) long, -1 = padding
            selected_mask:    (B, max_selected_frames) bool
            selected_count:   (B,) long, in [min_selected_frames, max_selected_frames]
            stop_step:        (B,) long, step where the episode ended (== selected_count;
                              equals max_selected_frames when force-stopped by the budget)
        """
        B, M, D = frame_features.shape
        device = frame_features.device
        H = self.hidden_size
        assert D == self.feature_dim, f"feature dim {D} != configured {self.feature_dim}"
        assert M <= self.max_candidates, f"M={M} exceeds policy head capacity {self.max_candidates}"
        assert M > self.max_selected_frames, \
            "need M > max_selected_frames so a force-stop cannot exhaust candidates"

   
        pos = self.memory_pos(torch.arange(M, device=device))           
        memory = torch.tanh(self.memory_proj(frame_features) + pos)     
        z0 = self.h0_proj(memory.mean(dim=1))                          
        h = self.h0_ln(z0)                                             
        c = torch.zeros(B, H, device=device)

     
        x_cur = frame_features.mean(dim=1)                           
        bar_x = x_cur.clone()

 
        selected = torch.full((B, self.max_selected_frames), -1, dtype=torch.long, device=device)
        sel_mask = torch.zeros(B, self.max_selected_frames, dtype=torch.bool, device=device)
        counts = torch.zeros(B, dtype=torch.long, device=device)
        stop_step = torch.full((B,), -1, dtype=torch.long, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)
        last_idx = torch.full((B,), -1, dtype=torch.long, device=device)
       
        action_log_probs = torch.zeros(B, self.max_selected_frames, device=device)
        hidden_states = torch.zeros(B, self.max_selected_frames, H, device=device)
        cand_valid = torch.ones(B, M, dtype=torch.bool, device=device) if frame_mask is None \
            else frame_mask.bool()

        j_range = torch.arange(M, device=device).unsqueeze(0)           # (1, M)

        for t in range(self.max_selected_frames):
   
            delta = frame_features - bar_x.unsqueeze(1)                # (B, M, D)
            interaction = frame_features * bar_x.unsqueeze(1)          # (B, M, D)
            pos_j = self.memory_pos(torch.arange(M, device=device))    # (M, H)
            cand_input = torch.cat(
                [frame_features,
                 bar_x.unsqueeze(1).expand(-1, M, -1),
                 delta, interaction,
                 pos_j.unsqueeze(0).expand(B, -1, -1)], dim=-1)        # (B, M, 4D+H)
            z_cand = self.candidate_mlp(cand_input)                    # (B, M, H)
            logits_frame = self.policy(
                torch.cat([h.unsqueeze(1).expand(-1, M, -1), z_cand], dim=-1)
            ).squeeze(-1)                                              # (B, M)
            logits = torch.cat([logits_frame, self.stop_head(h)], dim=-1)  # (B, M+1)

          
            forbid = torch.zeros(B, M + 1, dtype=torch.bool, device=device)
           
            if t > 0:
                forbid[:, :M] = j_range <= last_idx.unsqueeze(1)
           
            if t < self.min_selected_frames:
                forbid[:, :M] |= j_range > (M - self.min_selected_frames + t)
      
            forbid[:, :M] |= ~cand_valid
         
            forbid[:, M] = (t == 0) | (counts < self.min_selected_frames)

            masked_logits = logits.masked_fill(forbid, float("-inf"))

            action = torch.full((B,), -1, dtype=torch.long, device=device)
            if active.any():
                probs = torch.softmax(masked_logits[active], dim=-1)
                if self.sample:
                    dist = torch.distributions.Categorical(probs=probs)
                    action[active] = dist.sample()
                    action_log_probs[active, t] = dist.log_prob(action[active])
                else:
                    action[active] = probs.argmax(dim=-1)
                    action_log_probs[active, t] = torch.log(
                        probs.gather(1, action[active].unsqueeze(1)).squeeze(1) + 1e-12)

            is_stop = action == M
            just_stopped = active & is_stop
            just_selected = active & ~is_stop

        
            stop_step[just_stopped] = t
            active = active & ~just_stopped

       
            if just_selected.any():
              
                hidden_states[just_selected, t] = h[just_selected]
               
                safe_action = action.clamp(min=0, max=M - 1)
                x_new = frame_features[torch.arange(B, device=device), safe_action]   # (B, D)

          
                delta = x_new - bar_x
                hadamard = x_new * bar_x
                incremental = self.incremental_mlp(
                    torch.cat([x_new, bar_x, delta, hadamard], dim=-1))              # (B, H)

              
                t_ctx = self.step_pos(torch.full((B,), t, dtype=torch.long, device=device))
                _, (h_n, c_n) = self.lstm(
                    (incremental + t_ctx).unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
                h_new, c_new = h_n.squeeze(0), c_n.squeeze(0)

                upd = just_selected.unsqueeze(1)
                h = torch.where(upd, h_new, h)
                c = torch.where(upd, c_new, c)
                x_cur = torch.where(upd, x_new, x_cur)

                # running mean of selected features
                new_counts = (counts + 1).unsqueeze(1)
                bar_x = torch.where(
                    upd, (bar_x * counts.unsqueeze(1) + x_new) / new_counts, bar_x)

                selected[just_selected, t] = action[just_selected]
                sel_mask[just_selected, t] = True
                counts = torch.where(just_selected, counts + 1, counts)
                last_idx = torch.where(just_selected, safe_action, last_idx)

            if not active.any():
                break

        
        stop_step[active] = self.max_selected_frames

        return {
            "selected_indices": selected,
            "selected_mask": sel_mask,
            "selected_count": counts,
            "stop_step": stop_step,
            
            "action_log_probs": action_log_probs,   # (B, max_sel) log pi(a_t|h_t)
            "hidden_states": hidden_states,         # (B, max_sel, H) pre-update h_t
        }
