import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing full integration — your modules + Shreyashree's modules...")

# ---- your modules ----
from security.byzantine import ByzantineFaultInjector
from security.anomaly_detector import AnomalyDetector
from security.policy_checker import PolicyChecker
from coordination.consensus import ConsensusEngine
from coordination.trust import TrustManager
from coordination.state_sharing import StateSharing

# ---- Shreyashree's modules ----
from env.ran_env import RANEnv
from agents.mappo_agent import MAPPOAgent
from agents.agent_manager import AgentManager

print("✅ All imports successful")

# ---- test 1: environment setup ----
print("\n--- Test 1: Environment Setup ---")
try:
    env = RANEnv()
    obs, infos = env.reset()
    print(f"✅ Environment created")
    print(f"   Agents: {env.agents}")
    print(f"   Observation shape: {obs[env.agents[0]].shape}")
except Exception as e:
    print(f"❌ Environment failed: {e}")
    sys.exit(1)

# ---- test 2: agent setup ----
print("\n--- Test 2: Agent Setup ---")
try:
    n_agents = len(env.agents)
    obs_dim = obs[env.agents[0]].shape[0]
    act_dim = env.action_space(env.agents[0]).n

    print(f"   n_agents: {n_agents}")
    print(f"   obs_dim : {obs_dim}")
    print(f"   act_dim : {act_dim}")

    manager = AgentManager(
        num_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=act_dim
    )
    print("✅ AgentManager created")
except Exception as e:
    print(f"❌ AgentManager failed: {e}")
    sys.exit(1)

# ---- test 3: security + coordination setup ----
print("\n--- Test 3: Security + Coordination Setup ---")
injector = ByzantineFaultInjector(total_agents=n_agents)
injector.inject(agent_id=0, attack_type="random")

detector = AnomalyDetector(n_agents=n_agents, window_size=20, threshold=3.0)
checker = PolicyChecker(n_agents=n_agents, action_space_size=act_dim)
consensus = ConsensusEngine(n_agents=n_agents, action_space_size=act_dim)
trust = TrustManager(n_agents=n_agents)
state_sharing = StateSharing(n_agents=n_agents, state_dim=obs_dim)

print("✅ Security + Coordination modules ready")

# ---- test 4: full episode ----
print("\n--- Test 4: Full Episode (20 steps) ---")
try:
    obs, infos = env.reset()
    agent_list = env.agents

    for step in range(20):
        # get actions from agent manager
        # convert obs dict to numpy array in agent order
        obs_array = np.array([obs[agent] for agent in agent_list], dtype=np.float32)
        actions_raw, log_probs, values = manager.select_actions(obs_array)

        # convert to dict with int agent ids
        actions = {
            i: injector.get_action(i, int(actions_raw[i]), act_dim)
            for i, agent in enumerate(agent_list)
        }
        # share states
        states = {
            i: obs[agent]
            for i, agent in enumerate(agent_list)
        }
        state_sharing.broadcast_all(states)

        # detect anomalies
        flagged = detector.run_all_detectors(actions)

        # check policies
        policy_results = checker.validate_all(actions)

        # update trust
        trust.update_on_anomaly(flagged)
        trust.update_on_policy(policy_results)

        # reach consensus
        trust_weights = trust.get_trust_weights()
        final_action, agreement, ok = consensus.reach_consensus(
            actions,
            flagged_agents=flagged,
            trust_scores=trust_weights
        )

        # update trust post consensus
        if final_action is not None:
            trust.update_on_consensus(actions, final_action)

        # step environment with original actions
        env_actions = {
            agent: int(actions_raw[i])
            for i, agent in enumerate(agent_list)
        }
        obs, rewards, terminations, truncations, infos = env.step(env_actions)

        state_sharing.step()

        print(f"Step {step+1:2d}: final_action={final_action}, "
              f"agreement={agreement:.1%}, "
              f"consensus={'✓' if ok else '✗'}, "
              f"flagged={flagged}")

        # check if episode done
        if all(terminations.values()) or all(truncations.values()):
            print("Episode ended early")
            break

    print("\n✅ Full episode completed!")

except Exception as e:
    print(f"❌ Episode failed at step {step+1}: {e}")
    import traceback
    traceback.print_exc()

# ---- summaries ----
print()
trust.summary()
consensus.summary()
detector.summary()

print("🎉 Full integration test passed!")