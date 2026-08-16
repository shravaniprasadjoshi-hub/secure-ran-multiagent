import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

from security.byzantine import ByzantineFaultInjector
from security.anomaly_detector import AnomalyDetector
from security.policy_checker import PolicyChecker
from coordination.consensus import ConsensusEngine
from coordination.trust import TrustManager
from coordination.state_sharing import StateSharing

print("✅ All imports successful")

# ---- test 1: byzantine injection ----
print("\n--- Test 1: Byzantine Injection ---")
injector = ByzantineFaultInjector(total_agents=8)
injector.inject(agent_id=3, attack_type="gradual")
injector.inject(agent_id=6, attack_type="random")
injector.status()
print("✅ Byzantine injection works")

# ---- test 2: anomaly detection ----
print("\n--- Test 2: Anomaly Detection ---")
detector = AnomalyDetector(n_agents=8, window_size=20, threshold=2.0)

for step in range(25):
    actions = {i: np.random.randint(0, 3) for i in range(8)}
    actions[3] = np.random.randint(3, 5)
    actions[6] = np.random.randint(3, 5)
    flagged = detector.run_all_detectors(actions)

detector.summary()
print("✅ Anomaly detection works")

# ---- test 3: policy checker ----
print("\n--- Test 3: Policy Checker ---")
checker = PolicyChecker(n_agents=8, action_space_size=5)
actions = {i: np.random.randint(0, 5) for i in range(8)}
rsrp_map = {i: -85 for i in range(8)}
sinr_map = {i: 10 for i in range(8)}
results = checker.validate_all(actions, rsrp_map=rsrp_map, sinr_map=sinr_map)
checker.summary()
print("✅ Policy checker works")

# ---- test 4: trust manager ----
print("\n--- Test 4: Trust Manager ---")
trust = TrustManager(n_agents=8)
trust.update_on_anomaly(flagged_agents=[3, 6])
trust.update_on_policy(results)
trust.summary()
print("✅ Trust manager works")

# ---- test 5: consensus ----
print("\n--- Test 5: Consensus Engine ---")
consensus = ConsensusEngine(n_agents=8, action_space_size=5)
trust_weights = trust.get_trust_weights()
final_action, agreement, ok = consensus.reach_consensus(
    actions,
    flagged_agents=[3, 6],
    trust_scores=trust_weights
)
consensus.summary()
print("✅ Consensus works")

# ---- test 6: state sharing ----
print("\n--- Test 6: State Sharing ---")
state_sharing = StateSharing(n_agents=8, state_dim=10)
states = {i: np.random.randn(10) for i in range(8)}
state_sharing.broadcast_all(states)
global_state = state_sharing.get_global_state()
print(f"Global state shape: {global_state.shape}")
state_sharing.summary()
print("✅ State sharing works")

# ---- test 7: full pipeline ----
print("\n--- Test 7: Full Pipeline (10 steps) ---")
detector.reset()
trust.reset()
consensus.reset()
state_sharing.reset()

for step in range(10):
    actions = {i: np.random.randint(0, 5) for i in range(8)}
    actions[3] = injector.get_action(3, actions[3], 5)
    actions[6] = injector.get_action(6, actions[6], 5)

    states = {i: np.random.randn(10) for i in range(8)}
    state_sharing.broadcast_all(states)

    flagged = detector.run_all_detectors(actions)
    policy_results = checker.validate_all(actions)

    trust.update_on_anomaly(flagged)
    trust.update_on_policy(policy_results)

    trust_weights = trust.get_trust_weights()
    final_action, agreement, ok = consensus.reach_consensus(
        actions,
        flagged_agents=flagged,
        trust_scores=trust_weights
    )

    trust.update_on_consensus(actions, final_action)
    state_sharing.step()

    print(f"Step {step+1}: action={final_action}, "
          f"agreement={agreement:.1%}, ok={ok}, flagged={flagged}")

print("\n✅ Full pipeline works!")
print("\n🎉 All tests passed — integration successful!")