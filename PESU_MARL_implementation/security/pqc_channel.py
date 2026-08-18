"""
security/pqc_channel.py: Post-quantum secure channel between MARL agents and the coordinator
Owner: Shreyashree
Depends on: liboqs-python (import oqs - needs the liboqs C lib built via cmake/gcc), cryptography
Used by: training/train_secure.py (Exp 4), dashboard/sim_runner.py (Exp 4 live sim, once wired in)

# Exp 4 threat model addition: Exp 3's AnomalyDetector/ConsensusEngine trust that
# an agent's message really came from that agent, unmodified in transit - they
# only judge BEHAVIOR (drift, disagreement, statistical outliers). This module
# closes the transit gap: message spoofing/forgery/tampering, a DIFFERENT
# attack class than byzantine.py's behavioral attacks.
#
# Design:
#   - Kyber768 (ML-KEM) KEM: one handshake per agent per episode -> derives a
#     per-agent AES-256-GCM session key via HKDF. Confidentiality.
#   - Dilithium3 (ML-DSA) signatures: every per-step action proposal is signed
#     by the agent's long-term identity key before reaching ConsensusEngine /
#     AnomalyDetector. Authenticity + integrity.
#   - A tampered/forged message fails verification and is dropped BEFORE it
#     reaches consensus - caller should treat it like a flagged/quarantined
#     proposal (see receive_action() docstring).
#
# NOTE ON ALGORITHM NAMES: liboqs renamed Kyber768 -> ML-KEM-768 and
# Dilithium3 -> ML-DSA-65 in newer releases. _resolve_alg() below tries the
# legacy name first, falls back to the new name, and raises with the actual
# enabled list if neither exists - check that against whatever liboqs
# version you built.
#
# NOT WIRED INTO sim_runner.py YET - this is the standalone module only.
# Integration point (next step): in run_eval_episode(), each agent's raw
# action goes through channel.send_action() before detector.run_all_detectors()
# / consensus.reach_consensus(); only actions where receive_action() returns
# verified=True get passed through - forged/tampered ones get treated as
# flagged, same as a Byzantine-flagged agent.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field

import oqs
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

DEFAULT_KEM_ALG = "Kyber768"      # fallback: "ML-KEM-768" on newer liboqs
DEFAULT_SIG_ALG = "Dilithium3"    # fallback: "ML-DSA-65" on newer liboqs


def _resolve_alg(preferred: str, fallback: str, enabled: list) -> str:
    if preferred in enabled:
        return preferred
    if fallback in enabled:
        return fallback
    raise RuntimeError(
        f"neither {preferred!r} nor {fallback!r} available in this liboqs build - "
        f"enabled mechanisms: {enabled}"
    )


@dataclass
class PQCMetrics:
    """Running counters - printed via summary(), same convention as trust.py / consensus.py."""
    kem_handshakes: int = 0
    kem_time_total: float = 0.0
    sign_count: int = 0
    sign_time_total: float = 0.0
    verify_count: int = 0
    verify_time_total: float = 0.0
    verify_failures: int = 0        # tamper/forgery rejections - THE Exp 4 headline number
    encrypt_time_total: float = 0.0
    decrypt_time_total: float = 0.0

    @staticmethod
    def _avg_ms(total: float, count: int) -> float:
        return (total / count) * 1000 if count else 0.0

    def summary(self):
        print("\n=== PQC Channel Summary ===")
        print(f"  KEM handshakes            : {self.kem_handshakes}  (avg {self._avg_ms(self.kem_time_total, self.kem_handshakes):.2f} ms)")
        print(f"  Signatures issued         : {self.sign_count}  (avg {self._avg_ms(self.sign_time_total, self.sign_count):.2f} ms)")
        print(f"  Verifications             : {self.verify_count}  (avg {self._avg_ms(self.verify_time_total, self.verify_count):.2f} ms)")
        print(f"  Rejected (tamper/forgery) : {self.verify_failures}")
        print(f"============================\n")


@dataclass
class SecureMessage:
    """What actually crosses the 'channel' between an agent and the coordinator."""
    agent_id: int
    step: int
    ciphertext: bytes    # AES-256-GCM(payload) using this agent's session key
    nonce: bytes
    signature: bytes     # Dilithium3 signature over (ciphertext + nonce), by the agent's identity key


class AgentIdentity:
    """
    One per agent - long-term Dilithium3 signing identity. Created once at
    registration and kept alive so repeated .sign() calls reuse the same
    secret key (regenerating a keypair every step would make the "identity"
    meaningless - a forger could just make a fresh one too).
    # CRITICAL: call .close() when done or the underlying liboqs C context leaks.
    """

    def __init__(self, agent_id: int, sig_alg: str):
        self.agent_id = agent_id
        self._sig = oqs.Signature(sig_alg)
        self.public_key = self._sig.generate_keypair()

    def sign(self, message: bytes) -> bytes:
        return self._sig.sign(message)

    def close(self):
        self._sig.free()


class PQCChannel:
    """
    Coordinator-side orchestrator: owns the coordinator's Kyber768 KEM identity,
    the registry of agent public keys, and this episode's per-agent session keys.

    Usage per episode:
        channel = PQCChannel()
        for agent_id in range(n_agents):
            channel.register_agent(agent_id)     # once per agent - persists across episodes
            channel.establish_session(agent_id)  # once per episode - KEM handshake

        # per step, agent side:
        msg = channel.send_action(agent_id, step, action)
        # per step, coordinator side:
        action, verified = channel.receive_action(msg)
        if not verified:
            # tampered/forged - do NOT hand `action` to ConsensusEngine,
            # treat this agent's proposal for this step as flagged instead
            ...
    """

    def __init__(self, kem_alg: str = None, sig_alg: str = None):
        enabled_kem = oqs.get_enabled_kem_mechanisms()
        enabled_sig = oqs.get_enabled_sig_mechanisms()
        self.kem_alg = kem_alg or _resolve_alg(DEFAULT_KEM_ALG, "ML-KEM-768", enabled_kem)
        self.sig_alg = sig_alg or _resolve_alg(DEFAULT_SIG_ALG, "ML-DSA-65", enabled_sig)

        # coordinator's own KEM identity - one keypair, reused to decap every
        # agent's per-episode handshake ciphertext (standard KEM usage: one
        # recipient keypair, many senders)
        self._kem = oqs.KeyEncapsulation(self.kem_alg)
        self.coordinator_kem_public_key = self._kem.generate_keypair()

        self._identities: dict[int, AgentIdentity] = {}
        self._session_keys: dict[int, bytes] = {}
        self.metrics = PQCMetrics()

    # registration / handshake

    def register_agent(self, agent_id: int) -> bytes:
        """Creates (or reuses) an agent's long-term signing identity. Returns its public key."""
        if agent_id not in self._identities:
            self._identities[agent_id] = AgentIdentity(agent_id, self.sig_alg)
        return self._identities[agent_id].public_key

    def establish_session(self, agent_id: int) -> None:
        """
        Simulates the agent-side KEM encapsulation + coordinator-side
        decapsulation for one episode. Call once per agent per episode.
        # DONT TOUCH the encap/decap order - encapsulator (agent) runs first,
        # ciphertext then travels "over the wire" to the decapsulator (coordinator).
        """
        t0 = time.perf_counter()

        with oqs.KeyEncapsulation(self.kem_alg) as agent_kem:
            ciphertext, agent_shared_secret = agent_kem.encap_secret(self.coordinator_kem_public_key)

        coordinator_shared_secret = self._kem.decap_secret(ciphertext)
        assert coordinator_shared_secret == agent_shared_secret, "KEM mismatch - handshake broken"

        self._session_keys[agent_id] = self._derive_aes_key(coordinator_shared_secret, agent_id)

        self.metrics.kem_handshakes += 1
        self.metrics.kem_time_total += time.perf_counter() - t0

    @staticmethod
    def _derive_aes_key(shared_secret: bytes, agent_id: int) -> bytes:
        """HKDF-SHA256 -> 32 bytes for AES-256-GCM. info binds the key to this agent_id."""
        return HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None,
            info=f"pqc-channel-agent-{agent_id}".encode(),
        ).derive(shared_secret)

    # per-step send / receive

    def send_action(self, agent_id: int, step: int, action: int) -> SecureMessage:
        """Agent side: encrypt + sign one step's proposed action."""
        if agent_id not in self._session_keys:
            raise RuntimeError(f"agent {agent_id}: call establish_session() before send_action()")
        if agent_id not in self._identities:
            raise RuntimeError(f"agent {agent_id}: call register_agent() before send_action()")

        payload = json.dumps({"agent_id": agent_id, "step": step, "action": action}).encode()

        t0 = time.perf_counter()
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._session_keys[agent_id]).encrypt(nonce, payload, None)
        self.metrics.encrypt_time_total += time.perf_counter() - t0

        t1 = time.perf_counter()
        signature = self._identities[agent_id].sign(ciphertext + nonce)
        self.metrics.sign_count += 1
        self.metrics.sign_time_total += time.perf_counter() - t1

        return SecureMessage(agent_id=agent_id, step=step, ciphertext=ciphertext,
                              nonce=nonce, signature=signature)

    def receive_action(self, msg: SecureMessage) -> tuple[int | None, bool]:
        """
        Coordinator side: verify signature, then decrypt.
        Returns (action, verified). On failure returns (None, False) - the
        caller should treat this exactly like a flagged/quarantined proposal
        (e.g. exclude from ConsensusEngine.reach_consensus() same as an
        AnomalyDetector flag) rather than passing None through as an action.
        """
        identity = self._identities.get(msg.agent_id)
        if identity is None:
            self.metrics.verify_failures += 1
            return None, False

        t0 = time.perf_counter()
        with oqs.Signature(self.sig_alg) as verifier:
            ok = verifier.verify(msg.ciphertext + msg.nonce, msg.signature, identity.public_key)
        self.metrics.verify_count += 1
        self.metrics.verify_time_total += time.perf_counter() - t0

        if not ok:
            self.metrics.verify_failures += 1
            return None, False

        t1 = time.perf_counter()
        try:
            payload = AESGCM(self._session_keys[msg.agent_id]).decrypt(msg.nonce, msg.ciphertext, None)
        except Exception:
            self.metrics.verify_failures += 1
            return None, False
        self.metrics.decrypt_time_total += time.perf_counter() - t1

        action = json.loads(payload)["action"]
        return action, True

    # lifecycle

    def close(self):
        self._kem.free()
        for identity in self._identities.values():
            identity.close()

    def summary(self):
        self.metrics.summary()


# ── attack simulation, for proving the module actually rejects tampering ──

def simulate_tamper(msg: SecureMessage, tamper_type: str = "flip_action") -> SecureMessage:
    """
    Test helper - returns a corrupted copy of a SecureMessage. Feed the result
    into PQCChannel.receive_action() and confirm it comes back verified=False
    (metrics.verify_failures += 1). This is the Exp 4 headline result: a
    Byzantine-style attacker that also tries to spoof messages gets caught
    here, on top of whatever AnomalyDetector/ConsensusEngine already catch.

    tamper_type:
      - "flip_action" : corrupts one byte of the ciphertext (payload tampering)
      - "forge_sig"   : replaces the signature with random bytes (forgery, no valid key)
      - "replay"      : returns the message UNCHANGED - PQCChannel does not
                         reject replays by itself (a resent valid message still
                         verifies). Pair this with a seen-(agent_id, step) check
                         at the call site if replay protection is needed.
    """
    tampered = copy.deepcopy(msg)
    if tamper_type == "flip_action":
        b = bytearray(tampered.ciphertext)
        b[0] ^= 0xFF
        tampered.ciphertext = bytes(b)
    elif tamper_type == "forge_sig":
        tampered.signature = os.urandom(len(tampered.signature))
    elif tamper_type != "replay":
        raise ValueError(f"unknown tamper_type: {tamper_type}")
    return tampered


# ── smoke test - run directly once liboqs-python is built to sanity-check ──

if __name__ == "__main__":
    channel = PQCChannel()
    print(f"using KEM={channel.kem_alg}  SIG={channel.sig_alg}")

    for agent_id in range(7):
        channel.register_agent(agent_id)
        channel.establish_session(agent_id)

    # legit round-trip
    msg = channel.send_action(agent_id=3, step=1, action=2)
    action, ok = channel.receive_action(msg)
    print(f"legit message: action={action}  verified={ok}")
    assert ok and action == 2

    # tampered payload should be rejected
    tampered = simulate_tamper(msg, "flip_action")
    action, ok = channel.receive_action(tampered)
    print(f"tampered payload: action={action}  verified={ok}")
    assert not ok

    # forged signature should be rejected
    forged = simulate_tamper(msg, "forge_sig")
    action, ok = channel.receive_action(forged)
    print(f"forged signature: action={action}  verified={ok}")
    assert not ok

    channel.summary()
    channel.close()
    print("smoke test passed")