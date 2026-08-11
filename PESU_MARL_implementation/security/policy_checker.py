import numpy as np

class PolicyChecker:
    """
    Validates agent actions against a set of predefined RAN policies.
    Acts as a rule-based safety layer on top of the ML/RL agents.
    
    Checks:
    - action bounds     : action is within valid range
    - handover rules    : handover only triggered when signal is weak enough
    - resource limits   : resource allocation doesn't exceed cell capacity
    - rate of change    : agent isn't changing actions too rapidly (instability)
    - consistency       : agent's action matches its local observation
    """

    def __init__(self, n_agents, action_space_size, max_resource=100):
        """
        n_agents         : total number of agents
        action_space_size: valid actions are 0 to action_space_size-1
        max_resource     : maximum resource units a cell can allocate
        """
        self.n_agents = n_agents
        self.action_space_size = action_space_size
        self.max_resource = max_resource

        # stores last action per agent to check rate of change
        self.last_actions = {i: None for i in range(n_agents)}

        # violation log
        self.violations = []

    def check_action_bounds(self, agent_id, action):
        """
        Most basic check — is the action even a valid one?
        Returns True if valid, False if out of bounds.
        """
        valid = 0 <= action < self.action_space_size
        if not valid:
            self._log_violation(agent_id, "action_bounds", 
                f"Action {action} out of range [0, {self.action_space_size-1}]")
        return valid

    def check_handover_policy(self, agent_id, action, rsrp, sinr, 
                               rsrp_threshold=-90, sinr_threshold=5):
        """
        Handover should only be triggered when signal quality is poor.
        If an agent triggers handover when signal is fine, that's suspicious.
        
        action          : 1 = handover triggered, 0 = stay
        rsrp            : signal strength in dBm (below -90 = weak)
        sinr            : signal quality in dB (below 5 = poor)
        Returns True if policy satisfied, False if violated.
        """
        handover_triggered = (action == 1)
        signal_is_weak = (rsrp < rsrp_threshold) or (sinr < sinr_threshold)

        if handover_triggered and not signal_is_weak:
            self._log_violation(agent_id, "handover_policy",
                f"Handover triggered with good signal — RSRP={rsrp}, SINR={sinr}")
            return False
        return True

    def check_resource_limits(self, agent_id, allocated_resources):
        """
        Agent cannot allocate more resources than the cell has.
        Returns True if within limits, False if exceeded.
        """
        if allocated_resources > self.max_resource:
            self._log_violation(agent_id, "resource_limits",
                f"Allocated {allocated_resources} > max {self.max_resource}")
            return False
        return True

    def check_rate_of_change(self, agent_id, action, max_change=2):
        """
        Flags agents that are switching actions too rapidly.
        Rapid unexplained switching is a sign of instability or attack.
        Returns True if change is acceptable, False if too rapid.
        """
        last = self.last_actions[agent_id]
        self.last_actions[agent_id] = action

        if last is None:
            return True  # first action, nothing to compare

        change = abs(action - last)
        if change > max_change:
            self._log_violation(agent_id, "rate_of_change",
                f"Action changed by {change} in one step (max allowed: {max_change})")
            return False
        return True

    def check_consistency(self, agent_id, action, observation, 
                          expected_action_fn=None):
        """
        Optional: checks if agent's action makes sense given its observation.
        expected_action_fn: a function that takes observation and returns 
                            what action we'd expect — you define this based 
                            on domain knowledge.
        Returns True if consistent, False if inconsistent.
        """
        if expected_action_fn is None:
            return True  # skip if no rule defined

        expected = expected_action_fn(observation)
        if action != expected:
            self._log_violation(agent_id, "consistency",
                f"Action {action} inconsistent with observation "
                f"(expected {expected})")
            return False
        return True

    def validate(self, agent_id, action, observation=None, 
                 rsrp=None, sinr=None, allocated_resources=None):
        """
        Runs all applicable checks for one agent in one step.
        Returns:
            passed (bool)  : True if all checks passed
            violations (list): list of check names that failed
        """
        failed_checks = []

        # always check bounds
        if not self.check_action_bounds(agent_id, action):
            failed_checks.append("action_bounds")

        # check handover if signal info provided
        if rsrp is not None and sinr is not None:
            if not self.check_handover_policy(agent_id, action, rsrp, sinr):
                failed_checks.append("handover_policy")

        # check resource limits if provided
        if allocated_resources is not None:
            if not self.check_resource_limits(agent_id, allocated_resources):
                failed_checks.append("resource_limits")

        # always check rate of change
        if not self.check_rate_of_change(agent_id, action):
            failed_checks.append("rate_of_change")

        # check consistency if observation and rule provided
        if observation is not None:
            if not self.check_consistency(agent_id, action, observation):
                failed_checks.append("consistency")

        passed = len(failed_checks) == 0
        return passed, failed_checks

    def validate_all(self, actions: dict, observations=None, 
                     rsrp_map=None, sinr_map=None):
        """
        Validates all agents in one call.
        actions     : {agent_id: action}
        observations: {agent_id: observation} (optional)
        rsrp_map    : {agent_id: rsrp_value} (optional)
        sinr_map    : {agent_id: sinr_value} (optional)
        
        Returns:
            results: {agent_id: {"passed": bool, "violations": list}}
        """
        results = {}

        for agent_id, action in actions.items():
            obs = observations.get(agent_id) if observations else None
            rsrp = rsrp_map.get(agent_id) if rsrp_map else None
            sinr = sinr_map.get(agent_id) if sinr_map else None

            passed, violations = self.validate(
                agent_id, action, 
                observation=obs,
                rsrp=rsrp, 
                sinr=sinr
            )
            results[agent_id] = {
                "passed": passed,
                "violations": violations
            }

        return results

    def _log_violation(self, agent_id, check_name, detail):
        """Internal — logs a policy violation."""
        entry = {
            "agent_id": agent_id,
            "check": check_name,
            "detail": detail
        }
        self.violations.append(entry)
        print(f"[PolicyChecker] ⚠ Agent {agent_id} | {check_name}: {detail}")

    def get_violations(self):
        return self.violations

    def reset(self):
        """Resets state between episodes."""
        self.last_actions = {i: None for i in range(self.n_agents)}
        self.violations.clear()

    def summary(self):
        """Prints violation summary."""
        print("\n=== Policy Checker Summary ===")
        print(f"Total violations: {len(self.violations)}")
        by_agent = {}
        for v in self.violations:
            aid = v["agent_id"]
            by_agent.setdefault(aid, []).append(v["check"])
        for agent_id, checks in by_agent.items():
            print(f"  Agent {agent_id}: {checks}")
        print(f"==============================\n")