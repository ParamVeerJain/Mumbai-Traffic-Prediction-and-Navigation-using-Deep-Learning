# scripts/train_ppo.py
"""
Train the PPO agent on the loaded graph.
Run after generate_dummy.py.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from src.graph import load_graph
from src.routing.ppo import PPOAgent, RoutingEnv
from config import DATA_DIR

GRAPH_CACHE = os.path.join(DATA_DIR, "graph.pkl")
EPISODES    = 500


def main():
    G = load_graph(cache_path=GRAPH_CACHE)
    nodes = list(G.nodes())

    agent = PPOAgent()
    print(f"[train_ppo] Training for {EPISODES} episodes...")

    for ep in range(EPISODES):
        origin, dest = random.sample(nodes, 2)
        env   = RoutingEnv(G, origin, dest, departure_hour=random.randint(6,22))
        state = env.reset()
        ep_reward = 0.0

        for _ in range(200):
            nbrs    = env._valid_neighbours(env.current)
            n_valid = min(len(nbrs), 8)
            if n_valid == 0:
                break
            action, lp, val = agent.select_action(state, n_valid)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, lp, reward, val, done)
            state      = next_state
            ep_reward += reward
            if done:
                break

        loss = agent.update()
        if ep % 50 == 0:
            print(f"  Ep {ep:4d} | reward {ep_reward:7.1f} | loss {loss:.4f}")

    agent.save()
    print("[train_ppo] Done.")


if __name__ == "__main__":
    main()