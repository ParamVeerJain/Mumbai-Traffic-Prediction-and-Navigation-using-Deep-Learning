# src/routing/ppo.py
"""
PPO routing agent.

WHY PPO BEYOND A*
─────────────────
A* is optimal given the predicted travel-times, but it's stateless — it
treats each query independently and never improves from experience.

PPO adds three things A* can't do:

1. STOCHASTIC RESILIENCE — A* picks the single best path. PPO learns a
   *distribution* over actions, so it naturally hedges uncertainty.
   If Road X is 20% faster on average but 30% of the time it jams badly,
   PPO learns to avoid it; A* always picks it.

2. TEMPORAL CREDIT ASSIGNMENT — PPO sees delayed rewards (you only know
   total trip time at the end). It can learn "taking the ring road adds
   2 min now but saves 8 min later because the city-centre bottleneck is
   unpredictable." A* has no memory.

3. SOFT PREFERENCES — Reward can encode comfort, road familiarity, safety.
   You just change the reward function. A* would need hard constraints.

HOW IT WORKS HERE
─────────────────
State  : [hour_sin, hour_cos, day_sin, day_cos,
          normalised_distance_to_goal,
          current_congestion, current_speed_ratio,
          is_signal, is_toll,
          nbr_0_tt, nbr_1_tt, ..., nbr_K_tt]   → padded to K=8

Action : choose one of K neighbours (or stay = episode ends badly)

Reward : +100 if destination reached
         -travel_time_this_step / 60      (seconds → negative minutes)
         -5  * is_signal  (optional signal penalty)
         -10 * is_toll    (optional toll penalty)
         -50 if stuck (no valid neighbours)
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from config import (PPO_GAMMA, PPO_LR, PPO_CLIP_EPS,
                    PPO_EPOCHS, PPO_HIDDEN, PPO_CHECKPOINT)

MAX_NEIGHBOURS = 8
STATE_DIM      = 4 + 1 + 2 + 2 + MAX_NEIGHBOURS   # = 17


# ── Networks ──────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=MAX_NEIGHBOURS,
                 hidden=PPO_HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h      = self.shared(x)
        logits = self.actor(h)
        value  = self.critic(h)
        return logits, value


# ── Environment ───────────────────────────────────────────────────────────────
class RoutingEnv:
    def __init__(self, G, origin, destination,
                 departure_hour=8, mode="fastest",
                 signal_penalty=5.0, toll_penalty=10.0):
        self.G               = G
        self.origin          = origin
        self.destination     = destination
        self.departure_hour  = departure_hour
        self.mode            = mode
        self.signal_penalty  = signal_penalty
        self.toll_penalty    = toll_penalty

        # normalisation constants
        self._max_dist = self._graph_diameter()
        self._max_tt   = 3600.0    # 1 hour cap per step

    def reset(self):
        self.current   = self.origin
        self.step_count = 0
        self.visited   = {self.origin}
        return self._state()

    def step(self, action_idx):
        nbrs = self._valid_neighbours(self.current)

        if action_idx >= len(nbrs):
            # invalid action → small penalty, stay put
            return self._state(), -5.0, False, {}

        next_node = nbrs[action_idx]
        edge      = self.G[self.current][next_node]
        tt        = edge.get("travel_time_s", 60.0)
        signals   = edge.get("traffic_signals", False)
        toll      = edge.get("toll", False)

        reward = -(tt / 60.0)
        if signals: reward -= self.signal_penalty
        if toll:    reward -= self.toll_penalty

        self.current = next_node
        self.visited.add(next_node)
        self.step_count += 1
        done = (next_node == self.destination) or (self.step_count > 200)

        if next_node == self.destination:
            reward += 100.0

        return self._state(), reward, done, {"reached": next_node == self.destination}

    def _state(self):
        node = self.current
        hour = (self.departure_hour + self.step_count) % 24
        dow  = 0   # simplified
        nbrs = self._valid_neighbours(node)

        goal_dist = self._dist_to_goal(node) / (self._max_dist + 1e-6)

        edge_to_here = {}
        for p in self.G.predecessors(node):
            if self.G.has_edge(p, node):
                edge_to_here = self.G[p][node]
                break

        congestion = edge_to_here.get("speed_ratio", 0.5)
        sr         = 1.0 - congestion

        nbr_tts = [0.0] * MAX_NEIGHBOURS
        for i, nb in enumerate(nbrs[:MAX_NEIGHBOURS]):
            tt = self.G[node][nb].get("travel_time_s", 60.0)
            nbr_tts[i] = min(tt / self._max_tt, 1.0)

        is_signal = float(edge_to_here.get("traffic_signals", False))
        is_toll   = float(edge_to_here.get("toll", False))

        state = [
            math.sin(2*math.pi*hour/24),
            math.cos(2*math.pi*hour/24),
            math.sin(2*math.pi*dow/7),
            math.cos(2*math.pi*dow/7),
            goal_dist, congestion, sr,
            is_signal, is_toll,
        ] + nbr_tts

        return np.array(state, dtype=np.float32)

    def _valid_neighbours(self, node):
        nbrs = list(self.G.successors(node))
        if self.mode == "no_signals":
            nbrs = [n for n in nbrs
                    if not self.G[node][n].get("traffic_signals", False)]
        if self.mode == "no_toll":
            nbrs = [n for n in nbrs
                    if not self.G[node][n].get("toll", False)]
        return nbrs

    def _dist_to_goal(self, node):
        from src.graph import haversine_m
        return haversine_m(
            self.G.nodes[node]["lat"],       self.G.nodes[node]["lon"],
            self.G.nodes[self.destination]["lat"],
            self.G.nodes[self.destination]["lon"],
        )

    def _graph_diameter(self):
        from src.graph import haversine_m
        nodes = list(self.G.nodes(data=True))
        if len(nodes) < 2:
            return 1000.0
        lats = [d["lat"] for _, d in nodes]
        lons = [d["lon"] for _, d in nodes]
        return haversine_m(min(lats), min(lons), max(lats), max(lons))


# ── PPO Agent ─────────────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net    = ActorCritic().to(self.device)
        self.opt    = optim.Adam(self.net.parameters(), lr=PPO_LR)
        self.buffer = []
        if os.path.exists(PPO_CHECKPOINT):
            self.net.load_state_dict(torch.load(PPO_CHECKPOINT, map_location=self.device))
            print(f"[PPO] Loaded checkpoint {PPO_CHECKPOINT}")

    def select_action(self, state, n_valid):
        """Returns (action_idx, log_prob, value)."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(s)
        # mask out invalid actions
        mask = torch.full((1, MAX_NEIGHBOURS), float("-inf"), device=self.device)
        mask[0, :n_valid] = 0.0
        probs = torch.softmax(logits + mask, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, s, a, lp, r, v, done):
        self.buffer.append((s, a, lp, r, v, done))

    def update(self):
        if len(self.buffer) < 32:
            return 0.0
        states, actions, old_lps, rewards, values, dones = zip(*self.buffer)
        self.buffer.clear()

        # GAE returns
        returns = _compute_returns(rewards, values, dones, PPO_GAMMA)

        S  = torch.tensor(np.array(states),  dtype=torch.float32).to(self.device)
        A  = torch.tensor(actions,            dtype=torch.long   ).to(self.device)
        LP = torch.tensor(old_lps,            dtype=torch.float32).to(self.device)
        R  = torch.tensor(returns,            dtype=torch.float32).to(self.device)
        V  = torch.tensor(values,             dtype=torch.float32).to(self.device)
        ADV = (R - V).detach()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        total_loss = 0.0
        for _ in range(PPO_EPOCHS):
            logits, vals = self.net(S)
            dist     = torch.distributions.Categorical(torch.softmax(logits, -1))
            new_lp   = dist.log_prob(A)
            ratio    = torch.exp(new_lp - LP)

            actor_loss  = -torch.min(
                ratio * ADV,
                torch.clamp(ratio, 1-PPO_CLIP_EPS, 1+PPO_CLIP_EPS) * ADV
            ).mean()
            critic_loss = 0.5 * (vals.squeeze() - R).pow(2).mean()
            entropy     = dist.entropy().mean()
            loss        = actor_loss + critic_loss - 0.01 * entropy

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
            total_loss += loss.item()

        return total_loss / PPO_EPOCHS

    def route(self, G, origin, destination, departure_hour=8, mode="fastest"):
        """Greedy rollout using learned policy. Returns path."""
        env  = RoutingEnv(G, origin, destination, departure_hour, mode)
        state = env.reset()
        path  = [origin]
        for _ in range(200):
            nbrs    = env._valid_neighbours(env.current)
            n_valid = min(len(nbrs), MAX_NEIGHBOURS)
            if n_valid == 0:
                break
            action, _, _ = self.select_action(state, n_valid)
            state, reward, done, info = env.step(action)
            path.append(env.current)
            if done:
                break
        return path if path[-1] == destination else None

    def save(self):
        os.makedirs(os.path.dirname(PPO_CHECKPOINT), exist_ok=True)
        torch.save(self.net.state_dict(), PPO_CHECKPOINT)
        print(f"[PPO] Saved {PPO_CHECKPOINT}")


def _compute_returns(rewards, values, dones, gamma):
    returns = []
    R = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        R = r + gamma * R * (1.0 - float(d))
        returns.insert(0, R)
    return returns