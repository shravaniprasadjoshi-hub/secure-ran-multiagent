"""
Generate synthetic datasets for Secure Multi-Agent AI Framework for RAN Control Loops.
Outputs:
  - data/rag_corpus.jsonl          : RAG knowledge chunks
  - data/ran_multi_agent_telemetry_70k.csv : ~70k telemetry rows
  - data/dataset_metadata.json     : schema and generation metadata
"""

from __future__ import annotations

import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SEED = 42
NUM_ROWS = 70_000
DATA_DIR = Path(__file__).resolve().parent / "data"

AGENT_TYPES = [
    "mobility_agent",
    "security_agent",
    "resource_agent",
    "energy_agent",
    "trust_agent",
    "beamforming_agent",
    "qos_agent",
    "policy_agent",
]

SCENARIOS = {
    "normal_operation": 0.55,
    "jamming_attack": 0.10,
    "compromised_ai_agent": 0.08,
    "adversarial_mobility_attack": 0.07,
    "massive_ue_mobility": 0.12,
    "high_congestion": 0.08,
}

THREAT_TYPES = [
    "none",
    "ai_poisoning",
    "adversarial_input",
    "rogue_xapp",
    "signaling_storm",
    "sybil_agent",
    "jamming",
    "fake_ue_context",
    "model_tampering",
]

RRC_STATES = ["RRC_IDLE", "RRC_INACTIVE", "RRC_CONNECTED"]
RRC_ACTIONS = [
    "none",
    "handover_trigger",
    "handover_cancel",
    "rrc_reconfiguration",
    "beam_switch",
    "admission_accept",
    "admission_reject",
    "bearer_adaptation",
    "security_mitigation",
]

RIC_COMPONENTS = ["near_rt_ric", "non_rt_ric", "o_du", "o_cu", "smo", "e2_node"]

CSV_COLUMNS = [
    "timestamp",
    "ue_id",
    "cell_id",
    "gnb_id",
    "neighbor_cell_id",
    "agent_id",
    "agent_type",
    "ric_component",
    "rrc_state",
    "scenario_type",
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "cqi",
    "rssi_dbm",
    "beam_index",
    "neighbor_rsrp_dbm",
    "ue_speed_kmh",
    "ue_direction_deg",
    "ho_history_count",
    "prb_utilization_pct",
    "cell_load_pct",
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "spectral_efficiency_bps_hz",
    "interference_dbm",
    "trust_score",
    "attack_probability",
    "anomaly_score",
    "agent_confidence",
    "consensus_vote_pct",
    "policy_compliance_score",
    "ho_required",
    "target_cell_id",
    "threat_type",
    "allocated_prb_count",
    "rrc_action",
    "ho_success",
    "energy_saving_mode",
    "pqc_key_exchange_status",
]


def build_rag_corpus() -> list[dict]:
    """Build structured RAG knowledge chunks aligned with project references."""
    chunks: list[dict] = []

    def add(
        title: str,
        category: str,
        source: str,
        content: str,
        keywords: list[str],
        extra: dict | None = None,
    ) -> None:
        chunks.append(
            {
                "doc_id": str(uuid.uuid4()),
                "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                "title": title,
                "category": category,
                "source": source,
                "content": content.strip(),
                "keywords": keywords,
                "metadata": extra or {},
            }
        )

    # --- Project overview ---
    add(
        "Secure Multi-Agent AI Framework for RAN Control Loops - Overview",
        "project_overview",
        "NBUC Project Problem Statement / Swetha Kerahalli",
        """The Secure Multi-Agent AI Framework for RAN Control Loops enables autonomous and
        trustworthy control of RAN operations by embedding agentic intelligence within RRC
        decision loops. Distributed AI agents collaboratively monitor network states, detect
        security anomalies, and optimize control actions in real time. The framework addresses
        centralized AI single points of failure, adversarial AI manipulation of RRC decisions,
        and lack of standardized secure coordination among multiple AI entities in AI-native 6G RANs.""",
        ["multi-agent", "RRC", "O-RAN", "6G", "secure AI", "RAN control loops"],
        {"project_duration_months": 6, "domain": "AI-native 6G Open RAN"},
    )

    add(
        "Multi-Agent Layer Architecture",
        "architecture",
        "Project Architecture Document",
        """The Multi-Agent Layer contains specialized agents: Mobility Agent (handover optimization),
        Security Agent (attack detection), Resource Allocation Agent (spectrum/PRB optimization),
        Interference Management Agent, Energy Optimization Agent, Handover Prediction Agent,
        Beamforming Agent, QoS Agent, Policy Agent, and Trust Agent. Each agent operates within
        defined security and policy boundaries, performing local inference and exchanging trust
        scores, state vectors, and recommendations via secure message bus (Kafka/MQTT/gRPC/E2).""",
        ["mobility agent", "security agent", "resource agent", "trust agent", "MARL"],
    )

    add(
        "Consensus-Based RRC Decision Validation",
        "security",
        "Project Security Architecture",
        """Consensus engine validates multi-agent RRC proposals using Byzantine Fault Tolerance
        and weighted trust voting. Decision acceptance criteria: majority agreement >70%,
        trust score >0.8, confidence score >0.85, and low security risk. Validated actions
        trigger handover, RRC reconfiguration, bearer setup, beam switching, or admission updates.
        Compromised agents sending malicious HO recommendations are rejected; rogue agents are
        isolated while backup agents maintain RAN stability.""",
        ["consensus", "BFT", "trust voting", "RRC validation", "rogue agent"],
    )

    # --- 3GPP specifications ---
    specs = [
        (
            "TS 38.300 NR Overall Architecture",
            "3gpp_architecture",
            "3GPP TS 38.300 Rel-18/19",
            """TS 38.300 defines NR overall architecture including gNB, CU/DU functional split,
            NG-RAN interfaces, and bearer architecture. AI-native RAN control loops map to gNB-CU-CP
            for RRC decisions, gNB-CU-UP for bearer management, and gNB-DU for real-time scheduling.
            Multi-agent AI deployment spans CU, DU, and Near-RT RIC for distributed intelligence.""",
            ["NR architecture", "gNB", "CU-DU split", "NG-RAN"],
        ),
        (
            "TS 38.331 RRC Protocol and Mobility Events",
            "3gpp_rrc",
            "3GPP TS 38.331 Rel-18",
            """TS 38.331 specifies RRC protocol for NR including connection control, mobility,
            measurement configuration, and handover procedures. Key measurement events for mobility:
            Event A1 (serving better than threshold), A2 (serving worse than threshold),
            A3 (neighbor offset better than PCell), A4 (neighbor better than threshold),
            A5 (PCell worse than Thresh1 AND neighbor better than Thresh2), A6 (neighbor offset
            better than SCell). Variables Ms, Mn, Mp use RSRP in dBm or RSRQ/SS-SINR in dB.
            Handover interruption time must satisfy Tinterrupt constraints per TS 38.133.""",
            ["RRC", "handover", "Event A3", "measurement report", "mobility"],
        ),
        (
            "TS 38.215 Physical Layer Measurements",
            "3gpp_measurements",
            "3GPP TS 38.215 Rel-18",
            """TS 38.215 defines NR physical layer measurements at UE antenna connector.
            RSRP: reference signal received power for cell-specific signal strength.
            RSRQ: N x RSRP / RSSI ratio indicating signal quality.
            SS-SINR: signal-to-noise and interference ratio, range approximately -23 to 40 dB.
            CSI-RSRQ reporting range: -43 dB to 20 dB with 0.5 dB resolution.
            These measurements are primary input features for AI mobility and resource agents.""",
            ["RSRP", "RSRQ", "SINR", "CSI", "physical measurements"],
        ),
        (
            "TS 38.214 Physical Layer Procedures and Link Adaptation",
            "3gpp_phy",
            "3GPP TS 38.214 Rel-18",
            """TS 38.214 covers NR physical layer procedures including scheduling, HARQ,
            MCS selection, and link adaptation. CQI reported by UE guides DL scheduling.
            Resource Agent uses PRB allocation, MCS rates, and HARQ TB statistics for optimization.
            AI-assisted scheduling targets spectral efficiency and latency under QoS constraints.""",
            ["scheduling", "HARQ", "MCS", "link adaptation", "PRB"],
        ),
        (
            "TS 28.530 AI Management Services",
            "3gpp_ai_mgmt",
            "3GPP TS 28.530 Rel-18",
            """TS 28.530 defines AI/ML management services for network and service management.
            Covers AI model lifecycle: training, testing, deployment, inference, and monitoring.
            Supports intent-driven autonomous networking aligned with zero-touch operation.
            Multi-agent frameworks require AI model versioning, trust validation, and orchestration
            across Near-RT RIC (inference) and Non-RT RIC (training/policy).""",
            ["AI management", "ML lifecycle", "intent-driven", "zero-touch"],
        ),
        (
            "TS 28.105 Network Resource Model",
            "3gpp_nrm",
            "3GPP TS 28.105 Rel-18",
            """TS 28.105 specifies Network Resource Model (NRM) for 5GS management including
            AI/ML management capabilities. Defines information models for gNB, cells, beams,
            bearers, and AI/ML workflow steps: model training, testing, inference emulation,
            deployment, and inference execution. ML training may reside in management system
            while inference runs in NG-RAN node or Near-RT RIC.""",
            ["NRM", "AI/ML workflow", "model deployment", "inference"],
        ),
        (
            "TS 23.288 NWDAF Network Data Analytics",
            "3gpp_analytics",
            "3GPP TS 23.288 Rel-18/19",
            """TS 23.288 defines Network Data Analytics Function (NWDAF) architecture for
            5G analytics. Provides Analytics Accuracy Information and ML Model Accuracy Information.
            Supports Federated Learning where multiple AI entities collaboratively train models
            without centralizing raw data. Vertical Federated Learning handles different feature
            spaces across domains. NWDAF analytics feed multi-agent state awareness.""",
            ["NWDAF", "analytics", "federated learning", "ML accuracy"],
        ),
        (
            "TS 33.501 5G Security Architecture",
            "3gpp_security",
            "3GPP TS 33.501 Rel-18",
            """TS 33.501 defines 5G security architecture including authentication, key agreement,
            integrity protection, and privacy. Security Agent validates RRC actions against
            integrity policies. Zero-trust continuous verification protects agent communication.
            Post-quantum cryptography (CRYSTALS-Kyber, Dilithium) provides quantum-safe RRC control.""",
            ["5G security", "authentication", "integrity", "zero trust", "PQC"],
        ),
        (
            "TR 38.817 AI/ML for NR Air Interface",
            "3gpp_tr",
            "3GPP TR 38.817",
            """TR 38.817 studies AI/ML use cases for NR air interface including CSI feedback
            enhancement, beam management, and positioning. Primary reference for AI-native RAN
            evolution toward 6G. Supports dataset generation using channel models from TR 38.901.""",
            ["AI-native RAN", "air interface", "beam management", "CSI"],
        ),
        (
            "TR 23.700-80 AI/ML Architecture in 5G",
            "3gpp_tr",
            "3GPP TR 23.700-80",
            """TR 23.700-80 studies architecture enhancements for AI/ML in 5G Core and RAN.
            Defines AI/ML operation types, model distribution, and analytics integration.
            Relevant for multi-agent federated learning and secure model aggregation.""",
            ["AI/ML architecture", "5G", "model distribution"],
        ),
    ]
    for title, cat, src, content, kw in specs:
        add(title, cat, src, content, kw)

    # --- O-RAN ---
    oran_topics = [
        (
            "O-RAN Near-RT RIC and Control Loops",
            "o_ran",
            "O-RAN WG2 AIML / Near-RT RIC Architecture",
            """O-RAN defines three AI/ML control loops: Loop 1 (per-TTI scheduling in O-DU),
            Loop 2 (10-1000 ms in Near-RT RIC), Loop 3 (>1000 ms in Non-RT RIC for policies).
            Near-RT RIC hosts xApps for real-time optimization. Multi-agent inference deploys
            at Near-RT RIC with E2 interface latency 10 ms to 1 s. Non-RT RIC serves as
            federated learning central server with Near-RT RICs as distributed AI entities.""",
            ["Near-RT RIC", "xApp", "control loop", "E2 latency"],
        ),
        (
            "O-RAN E2 Interface KPIs and Measurements",
            "o_ran",
            "O-RAN WG3 E2GAP / UCR",
            """E2 interface reports UE-specific DL L1 measurements: PMI, RI, CQI, CSI-RSRP,
            CSI-SINR, SS-RSRP. Cell-level KPIs: active UE count, packet loss rate, DL packet drop
            rate, DRB RLC buffer occupancy, QoS flow count, HARQ TB MCS distribution, PRACH
            correlation per SSB beam index. Near-RT RIC uses E2 for UE/cell control messages
            and policy enforcement toward O-DU/O-CU.""",
            ["E2", "KPI", "CQI", "packet loss", "PRB", "beam index"],
        ),
        (
            "O-RAN xApp and rApp Framework",
            "o_ran",
            "O-RAN xApp/rApp Framework",
            """xApps run on Near-RT RIC for real-time RAN optimization (mobility, security,
            resource allocation). rApps run on Non-RT RIC for long-term analytics and policy training.
            Mobility Agent maps to mobility optimization xApp; Security Agent to threat mitigation xApp.
            Rogue xApps pose control manipulation risk; trust engine monitors xApp behavior.""",
            ["xApp", "rApp", "RIC", "mobility optimization"],
        ),
    ]
    for title, cat, src, content, kw in oran_topics:
        add(title, cat, src, content, kw)

    # --- MARL ---
    marl_topics = [
        (
            "MARL Algorithms for RAN Control",
            "marl",
            "Multi-Agent RL Literature",
            """Centralized Training Distributed Execution (CTDE) enables scalable multi-agent RAN control.
            Recommended algorithms: MADDPG (cooperative control), QMIX (joint optimization), PPO (stable
            learning), DQN (resource allocation), MAPPO (multi-agent optimization). Agents receive
            rewards for successful HO, low latency, stable throughput; penalties for dropped calls,
            false alarms, failed handovers. Coordination via shared state exchange and policy synchronization.""",
            ["MADDPG", "MAPPO", "QMIX", "CTDE", "reinforcement learning"],
        ),
        (
            "AI/ML Models per Agent Function",
            "ai_models",
            "Project AI Model Mapping",
            """Mobility prediction: LSTM, Transformer. Security detection: Autoencoder, CNN for jamming.
            Resource allocation: Deep RL, DQN. Trust evaluation: Graph Neural Networks (GNN).
            Traffic prediction: GRU. Policy arbitration: PPO. Agent coordination: MARL frameworks
            (Ray RLlib, PettingZoo). Federated learning with differential privacy for secure collaboration.""",
            ["LSTM", "autoencoder", "GNN", "Deep RL", "Transformer"],
        ),
    ]
    for title, cat, src, content, kw in marl_topics:
        add(title, cat, src, content, kw)

    # --- Security ---
    security_topics = [
        (
            "Threat Model for AI-Native RAN",
            "security",
            "Project Threat Model / NIST AI RMF",
            """Attack vectors: AI poisoning (wrong RRC decisions), adversarial inputs (false predictions),
            rogue xApps (control manipulation), signaling storms (network overload), Sybil attacks
            (fake agents), model stealing, jamming (mobility failure), fake UE context manipulation.
            Security mechanisms: dynamic trust scoring, Byzantine consensus, federated learning security,
            blockchain audit logging, zero-trust verification, post-quantum cryptography.""",
            ["AI poisoning", "adversarial", "rogue xApp", "jamming", "Sybil"],
        ),
        (
            "Quantum-Safe Security for Agent Communication",
            "quantum_security",
            "NIST PQC Standards",
            """Quantum-safe components: CRYSTALS-Kyber (key encapsulation), CRYSTALS-Dilithium
            (digital signatures), QKD for secure key exchange, quantum RNG for key generation.
            Protects inter-agent communication and RRC control signaling against future quantum
            decryption attacks. PQC key exchange status monitored per agent transaction.""",
            ["Kyber", "Dilithium", "QKD", "post-quantum", "quantum RNG"],
        ),
        (
            "Trust Engine and Anomaly Detection",
            "security",
            "Project Security Layer",
            """Trust Engine evaluates agent behavior, historical reliability, and anomaly frequency
            using Bayesian inference. Anomaly Detection Engine (autoencoder-based) detects malicious AI,
            abnormal signaling, rogue decisions, and adversarial patterns. Trust scores range 0-1;
            threshold 0.8 required for consensus participation. Security mitigation target latency <20 ms.""",
            ["trust score", "anomaly detection", "autoencoder", "Bayesian"],
        ),
    ]
    for title, cat, src, content, kw in security_topics:
        add(title, cat, src, content, kw)

    # --- Workflow stages ---
    for i, (stage, desc) in enumerate(
        [
            ("RAN Environment Initialization", "Initialize gNBs, O-RU/O-DU/O-CU, Near-RT/Non-RT RIC, xApps, core network (AMF/SMF/UPF), and UEs with carrier frequency, bandwidth, numerology, beamforming, and security policies."),
            ("Data Collection & Monitoring", "Collect SINR, RSRP, RSRQ, CQI, PRB utilization, throughput, latency, trust score, anomaly score, and agent confidence at 10-100 ms intervals from UE through gNB to RIC."),
            ("AI Agent Deployment", "Deploy mobility, security, resource, energy, trust, beamforming, QoS, and policy agents at edge cloud, Near-RT RIC, O-DU, and Non-RT RIC."),
            ("Local AI Inference", "Each agent performs feature extraction, model inference, prediction, and confidence evaluation independently on local telemetry."),
            ("Multi-Agent Coordination", "Agents exchange trust scores, local observations, confidence values, attack alerts, and mobility forecasts via Kafka/MQTT/gRPC/E2."),
            ("Security & Trust Validation", "Trust evaluation, 3GPP policy verification, and threat analysis before consensus."),
            ("Consensus-Based Decision Making", "BFT and weighted trust voting with >70% majority, trust >0.8, confidence >0.85."),
            ("RRC Control Execution", "Validated decisions sent via Near-RT RIC E2 to O-DU/O-CU/gNB/UE. HO decision <50 ms, scheduling <10 ms."),
            ("Network Feedback Monitoring", "Monitor throughput, packet loss, latency, HO success rate, trust stability for RL rewards."),
            ("Continuous Learning & Adaptation", "Federated learning, online learning, and MARL model updates at Non-RT RIC."),
        ],
        start=1,
    ):
        add(
            f"Workflow Stage {i}: {stage}",
            "workflow",
            "Project End-to-End Workflow",
            desc,
            ["workflow", stage.lower().replace(" ", "_"), "RAN control"],
            {"stage_number": i},
        )

    # --- Experimental scenarios ---
    for scenario, metrics in [
        ("Normal Operation", "Evaluate throughput, HO success rate, latency, spectral efficiency, trust stability."),
        ("Jamming Attack", "Evaluate resilience, jamming detection rate, mitigation speed, handover success under interference."),
        ("Compromised AI Agent", "Evaluate trust detection, consensus robustness, rogue agent isolation, service continuity."),
        ("Adversarial Mobility Attack", "Evaluate false HO trigger detection, RRC integrity, policy compliance."),
        ("Massive UE Mobility", "Evaluate MARL coordination scalability, HO prediction accuracy, cell load balancing."),
        ("High Congestion", "Evaluate PRB allocation, QoS degradation, resource agent optimization, latency under load."),
    ]:
        add(
            f"Experimental Scenario: {scenario}",
            "scenarios",
            "Project Evaluation Scenarios",
            f"In scenario '{scenario}', the multi-agent framework is evaluated for {metrics}",
            ["scenario", scenario.lower().replace(" ", "_"), "evaluation"],
        )

    # --- Measurement thresholds (from 3GPP) ---
    add(
        "3GPP Measurement Value Ranges and HO Thresholds",
        "3gpp_measurements",
        "TS 38.331 / TS 38.215 / TS 28.552",
        """Typical NR measurement ranges: RSRP from -156 dBm upward; RSRQ -43 to 20 dB;
        SS-SINR -23 to 40 dB; CQI 0-15. Event A2 triggers when serving RSRP + Hys < Thresh
        (typically -110 to -100 dBm). Event A3 triggers neighbor better than serving by Off (1-6 dB).
        HO decision latency target <50 ms. PRB utilization 0-100%. Latency KPI: 1-100 ms for URLLC/eMBB.
        Packet loss <1% normal, >5% under attack/congestion.""",
        ["threshold", "RSRP range", "A2", "A3", "KPI"],
    )

    # --- Evaluation metrics ---
    add(
        "Evaluation Metrics for Multi-Agent RAN Framework",
        "evaluation",
        "Project Evaluation Plan",
        """AI metrics: accuracy, precision, recall, F1-score, reward convergence.
        Network metrics: throughput, spectral efficiency, HO success rate, packet delivery ratio, latency.
        Security metrics: attack detection accuracy, false alarm rate, trust stability, recovery time.
        6G metrics: reliability, energy efficiency, AI robustness, resilience score.
        Consensus stability and trust evolution tracked over experimental scenarios.""",
        ["KPI", "F1-score", "HO success", "detection rate", "resilience"],
    )

    # --- FAQ-style Q&A chunks for RAG ---
    faqs = [
        (
            "What is the role of the Mobility Agent?",
            "The Mobility Agent predicts handover need, target cell, and HO timing using LSTM/Transformer models on RSRP, RSRQ, SINR, UE speed, and HO history. It proposes RRC handover triggers validated by consensus.",
        ),
        (
            "How does consensus prevent rogue agents from controlling the RAN?",
            "The consensus engine uses Byzantine Fault Tolerance and weighted trust voting. Decisions require >70% agent agreement, trust score >0.8, and confidence >0.85. Malicious HO recommendations from compromised agents are rejected and the agent is isolated.",
        ),
        (
            "What telemetry features are collected for AI agents?",
            "Radio: SINR, RSRP, RSRQ, CQI, RSSI, beam index. Mobility: UE speed, direction, HO history. Network: PRB utilization, cell load, throughput, latency, packet loss. Security: trust score, attack probability, anomaly score. AI: agent confidence per decision.",
        ),
        (
            "How does O-RAN E2 interface support multi-agent control?",
            "E2 connects Near-RT RIC to O-DU/O-CU E2 nodes with 10 ms to 1 s latency. It carries UE measurements (CQI, RSRP, SINR), cell KPIs, and control messages for handover, scheduling, and policy enforcement.",
        ),
        (
            "What attacks does the Security Agent detect?",
            "AI poisoning, adversarial inputs, rogue xApps, signaling storms, Sybil fake agents, jamming, fake UE context manipulation, and model tampering using autoencoder anomaly detection and trust scoring.",
        ),
    ]
    for q, a in faqs:
        add(
            q,
            "faq",
            "Project Knowledge Base",
            f"Question: {q}\nAnswer: {a}",
            q.lower().split()[:5],
            {"format": "qa"},
        )

    # --- Per-agent knowledge chunks ---
    agent_details = {
        "mobility_agent": (
            "Predicts handover need using RSRP/RSRQ/SINR trends, UE speed, HO history. "
            "Proposes target cell and HO timing. Uses LSTM/Transformer. Latency target <50 ms. "
            "Coordinates with beamforming and QoS agents before consensus."
        ),
        "security_agent": (
            "Monitors trust scores, anomaly patterns, signaling rates. Detects AI poisoning, "
            "rogue xApps, jamming, Sybil agents via autoencoder. Triggers security_mitigation "
            "RRC action. False alarm rate target <5%."
        ),
        "resource_agent": (
            "Optimizes PRB allocation (5-100 PRBs), scheduling, beam allocation. Uses Deep RL/DQN. "
            "Inputs: PRB utilization, cell load, CQI, throughput. Targets spectral efficiency."
        ),
        "energy_agent": (
            "Manages power optimization and energy saving modes (normal/deep_sleep/mimo_reduce). "
            "Balances energy efficiency vs QoS. Coordinates with resource agent on PRB throttling."
        ),
        "trust_agent": (
            "Computes dynamic trust scores (0-1) using GNN over agent interaction graph. "
            "Bayesian inference on historical reliability. Threshold 0.8 for consensus voting weight."
        ),
        "beamforming_agent": (
            "Predicts optimal beam index (0-63 SSB beams) from CSI-RSRP, CSI-SINR, UE location. "
            "Proposes beam_switch RRC actions. Uses CNN/Transformer on beam measurement history."
        ),
        "qos_agent": (
            "Ensures SLA compliance: latency <20 ms URLLC, throughput GBR. Proposes bearer_adaptation. "
            "Monitors packet loss, DRB buffer occupancy from E2 KPIs."
        ),
        "policy_agent": (
            "Validates 3GPP TS 38.331 RRC constraints and operator policies. Computes "
            "policy_compliance_score. Rejects non-compliant agent proposals before consensus."
        ),
    }
    for agent, desc in agent_details.items():
        add(
            f"{agent.replace('_', ' ').title()} - Function and Model",
            "agents",
            "Project Multi-Agent Architecture",
            desc,
            [agent, "agent", "RRC", "AI inference"],
            {"agent_type": agent},
        )

    # --- Additional 3GPP RRC event details ---
    for event, formula in [
        ("Event A1", "Ms - Hys > Thresh: Serving cell becomes better than threshold. Used to cancel HO."),
        ("Event A2", "Ms + Hys < Thresh: Serving becomes worse. Typical Thresh -110 to -100 dBm. HO preparation."),
        ("Event A3", "Mn + Ofn + Ocn - Hys > Mp + Ofp + Ocp + Off: Neighbor offset better than PCell."),
        ("Event A4", "Mn + Ofn + Ocn - Hys > Thresh: Neighbor becomes better than absolute threshold."),
        ("Event A5", "Mp + Hys < Thresh1 AND Mn + Ofn + Ocn - Hys > Thresh2: Dual threshold HO trigger."),
        ("Event A6", "Mn + Ocn - Hys > Ms + Ocs + Off: Neighbor offset better than SCell for CA."),
    ]:
        add(
            f"RRC Measurement {event}",
            "3gpp_rrc",
            "3GPP TS 38.331",
            f"{event} entry condition: {formula} Ms/Mn/Mp in dBm (RSRP) or dB (RSRQ/SINR).",
            [event, "measurement event", "handover trigger", "RRC"],
        )

    # --- RRC procedures ---
    for proc, detail in [
        ("Handover Execution", "RRCReconfiguration with mobilityControlInfo. UE ready on new UL PRACH within Dhandover ms. Interruption time per TS 38.133."),
        ("RRC Setup", "RRCSetup message after RRCSetupRequest. Establishes SRB1, security, measurement configuration."),
        ("RRC Reconfiguration", "Modifies bearer config, measurement gaps, beam config, dedicated NAS. Used for QoS adaptation."),
        ("Cell Reselection", "Idle/inactive UE selects cell based on S-criteria and R-criteria ranking."),
        ("Connection Resume", "RRC_INACTIVE to RRC_CONNECTED via RRCResumeRequest. Reduces signaling overhead."),
    ]:
        add(f"RRC Procedure: {proc}", "3gpp_rrc", "3GPP TS 38.331", detail, ["RRC", proc.lower(), "procedure"])

    # --- Simulation tools ---
    for tool, purpose in [
        ("MATLAB 5G Toolbox", "PHY/RAN simulation, channel models, NR waveform generation."),
        ("ns-3", "End-to-end network simulation with EPC and NR modules."),
        ("OpenAirInterface", "Open-source O-RAN gNB/DU/CU implementation."),
        ("srsRAN", "Software gNB/UE for RAN prototyping."),
        ("Ray RLlib", "MARL training framework for multi-agent coordination."),
        ("PettingZoo", "Multi-agent RL environment API."),
        ("Qiskit", "Quantum circuit simulation for PQC and QKD protocols."),
    ]:
        add(f"Simulation Tool: {tool}", "simulation", "Project Technology Stack", purpose, [tool, "simulation", "tool"])

    return chunks


def assign_scenario(rng: np.random.Generator, n: int) -> np.ndarray:
    labels = list(SCENARIOS.keys())
    probs = np.array(list(SCENARIOS.values()))
    return rng.choice(labels, size=n, p=probs)


def generate_telemetry_csv(path: Path, num_rows: int = NUM_ROWS) -> dict:
    rng = np.random.default_rng(SEED)
    start_time = datetime(2026, 4, 10, 0, 0, 0)

    n_cells = 48
    n_gnbs = 12
    cell_ids = np.arange(1, n_cells + 1)
    gnb_ids = np.repeat(np.arange(1, n_gnbs + 1), n_cells // n_gnbs)

    scenarios = assign_scenario(rng, num_rows)
    ue_ids = rng.integers(10000, 99999, size=num_rows)
    cell_idx = rng.integers(0, n_cells, size=num_rows)
    cell_id = cell_ids[cell_idx]
    gnb_id = gnb_ids[cell_idx]

    # Neighbor cell (different from serving)
    neighbor_offset = rng.choice([-1, 1, 2, -2], size=num_rows)
    neighbor_cell_id = np.clip(cell_id + neighbor_offset, 1, n_cells)
    same_mask = neighbor_cell_id == cell_id
    neighbor_cell_id[same_mask] = np.clip(cell_id[same_mask] + 1, 1, n_cells)

    agent_type_idx = rng.integers(0, len(AGENT_TYPES), size=num_rows)
    agent_types = np.array(AGENT_TYPES)[agent_type_idx]
    agent_ids = np.array([f"agent_{AGENT_TYPES[i][:4]}_{rng.integers(1,5)}" for i in agent_type_idx])

    ric_idx = rng.integers(0, len(RIC_COMPONENTS), size=num_rows)
    ric_components = np.array(RIC_COMPONENTS)[ric_idx]

    rrc_states = rng.choice(RRC_STATES, size=num_rows, p=[0.1, 0.15, 0.75])

    # Base radio measurements
    rsrp = rng.normal(-95, 8, num_rows)
    rsrq = rng.normal(-12, 4, num_rows)
    sinr = rng.normal(12, 6, num_rows)
    cqi = np.clip(rng.integers(1, 16, size=num_rows) + (sinr / 5).astype(int), 1, 15)
    rssi = rsrp + rng.normal(10, 3, num_rows)
    beam_index = rng.integers(0, 64, size=num_rows)
    neighbor_rsrp = rsrp + rng.normal(3, 5, num_rows)

    ue_speed = np.abs(rng.normal(30, 25, num_rows))
    ue_direction = rng.uniform(0, 360, size=num_rows)
    ho_history = rng.poisson(0.5, num_rows).astype(int)

    prb_util = np.clip(rng.beta(2, 5, num_rows) * 100, 0, 100)
    cell_load = np.clip(prb_util + rng.normal(0, 10, num_rows), 0, 100)
    dl_tput = np.clip(rng.lognormal(2.5, 0.6, num_rows), 0.1, 500)
    ul_tput = dl_tput * rng.uniform(0.1, 0.4, num_rows)
    latency = np.clip(rng.lognormal(2.8, 0.5, num_rows), 1, 200)
    pkt_loss = np.clip(rng.beta(1.5, 50, num_rows) * 10, 0, 100)
    spec_eff = np.clip(sinr / 30 + rng.normal(0.5, 0.2, num_rows), 0.1, 8)
    interference = rng.normal(-105, 10, num_rows)

    trust = np.clip(rng.beta(8, 2, num_rows), 0, 1)
    attack_prob = np.clip(rng.beta(1.5, 15, num_rows), 0, 1)
    anomaly = np.clip(rng.beta(2, 10, num_rows), 0, 1)
    confidence = np.clip(rng.beta(7, 2, num_rows), 0, 1)
    consensus = np.clip(rng.beta(6, 3, num_rows) * 100, 0, 100)
    policy = np.clip(rng.beta(9, 1, num_rows), 0, 1)

    ho_required = np.zeros(num_rows, dtype=int)
    threat_type = np.full(num_rows, "none", dtype=object)
    rrc_action = np.full(num_rows, "none", dtype=object)
    ho_success = np.ones(num_rows, dtype=int)
    allocated_prb = rng.integers(5, 50, size=num_rows)
    target_cell = cell_id.copy()
    energy_mode = np.full(num_rows, "normal", dtype=object)
    pqc_status = np.full(num_rows, "success", dtype=object)

    # Scenario-specific perturbations
    for i in range(num_rows):
        sc = scenarios[i]
        if sc == "jamming_attack":
            sinr[i] -= rng.uniform(8, 20)
            interference[i] += rng.uniform(10, 25)
            rsrp[i] -= rng.uniform(5, 15)
            attack_prob[i] = min(1.0, attack_prob[i] + rng.uniform(0.4, 0.6))
            anomaly[i] = min(1.0, anomaly[i] + rng.uniform(0.3, 0.5))
            threat_type[i] = "jamming"
            pkt_loss[i] = min(100, pkt_loss[i] + rng.uniform(5, 20))
            rrc_action[i] = rng.choice(["security_mitigation", "handover_trigger"])
            ho_required[i] = 1
            target_cell[i] = neighbor_cell_id[i]
        elif sc == "compromised_ai_agent":
            trust[i] = rng.uniform(0.1, 0.45)
            confidence[i] = rng.uniform(0.3, 0.6)
            consensus[i] = rng.uniform(20, 55)
            attack_prob[i] = min(1.0, attack_prob[i] + rng.uniform(0.5, 0.8))
            anomaly[i] = min(1.0, anomaly[i] + rng.uniform(0.4, 0.7))
            threat_type[i] = rng.choice(["ai_poisoning", "rogue_xapp", "model_tampering"])
            rrc_action[i] = "none"  # rejected by consensus
            ho_required[i] = 0
        elif sc == "adversarial_mobility_attack":
            ho_required[i] = 1
            neighbor_rsrp[i] = rsrp[i] - rng.uniform(10, 20)  # false HO trigger attempt
            attack_prob[i] = min(1.0, attack_prob[i] + rng.uniform(0.3, 0.5))
            threat_type[i] = rng.choice(["adversarial_input", "fake_ue_context"])
            rrc_action[i] = rng.choice(["none", "handover_cancel"])
            ho_success[i] = 0
        elif sc == "massive_ue_mobility":
            ue_speed[i] = rng.uniform(60, 150)
            ho_history[i] = rng.integers(3, 15)
            ho_required[i] = int(neighbor_rsrp[i] > rsrp[i] + 3)
            if ho_required[i]:
                target_cell[i] = neighbor_cell_id[i]
                rrc_action[i] = "handover_trigger"
        elif sc == "high_congestion":
            prb_util[i] = rng.uniform(75, 99)
            cell_load[i] = rng.uniform(80, 100)
            latency[i] = rng.uniform(30, 150)
            dl_tput[i] *= rng.uniform(0.2, 0.5)
            pkt_loss[i] = min(100, pkt_loss[i] + rng.uniform(2, 8))
            allocated_prb[i] = rng.integers(40, 100)
        else:  # normal
            ho_required[i] = int(neighbor_rsrp[i] > rsrp[i] + rng.uniform(2, 5))
            if ho_required[i]:
                target_cell[i] = neighbor_cell_id[i]
                rrc_action[i] = "handover_trigger"
            elif rsrp[i] < -110:
                rrc_action[i] = rng.choice(["rrc_reconfiguration", "beam_switch"])

    timestamps = [start_time + timedelta(seconds=int(s)) for s in rng.integers(0, 86400 * 30, num_rows)]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for i in range(num_rows):
            writer.writerow(
                [
                    timestamps[i].isoformat(),
                    f"UE_{ue_ids[i]}",
                    f"CELL_{cell_id[i]:03d}",
                    f"GNB_{gnb_id[i]:02d}",
                    f"CELL_{neighbor_cell_id[i]:03d}",
                    agent_ids[i],
                    agent_types[i],
                    ric_components[i],
                    rrc_states[i],
                    scenarios[i],
                    round(float(rsrp[i]), 2),
                    round(float(rsrq[i]), 2),
                    round(float(sinr[i]), 2),
                    int(cqi[i]),
                    round(float(rssi[i]), 2),
                    int(beam_index[i]),
                    round(float(neighbor_rsrp[i]), 2),
                    round(float(ue_speed[i]), 2),
                    round(float(ue_direction[i]), 2),
                    int(ho_history[i]),
                    round(float(prb_util[i]), 2),
                    round(float(cell_load[i]), 2),
                    round(float(dl_tput[i]), 2),
                    round(float(ul_tput[i]), 2),
                    round(float(latency[i]), 2),
                    round(float(pkt_loss[i]), 4),
                    round(float(spec_eff[i]), 3),
                    round(float(interference[i]), 2),
                    round(float(trust[i]), 4),
                    round(float(attack_prob[i]), 4),
                    round(float(anomaly[i]), 4),
                    round(float(confidence[i]), 4),
                    round(float(consensus[i]), 2),
                    round(float(policy[i]), 4),
                    int(ho_required[i]),
                    f"CELL_{target_cell[i]:03d}",
                    threat_type[i],
                    int(allocated_prb[i]),
                    rrc_action[i],
                    int(ho_success[i]),
                    energy_mode[i],
                    pqc_status[i],
                ]
            )

    return {
        "num_rows": num_rows,
        "num_columns": len(CSV_COLUMNS),
        "columns": CSV_COLUMNS,
        "scenario_distribution": {k: int((scenarios == k).sum()) for k in SCENARIOS},
        "file": str(path.name),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # RAG corpus
    corpus = build_rag_corpus()
    rag_path = DATA_DIR / "rag_corpus.jsonl"
    with rag_path.open("w", encoding="utf-8") as f:
        for chunk in corpus:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Telemetry CSV
    csv_path = DATA_DIR / "ran_multi_agent_telemetry_70k.csv"
    csv_meta = generate_telemetry_csv(csv_path, NUM_ROWS)

    # Metadata
    metadata = {
        "project": "Secure Multi-Agent AI Framework for RAN Control Loops",
        "generated_at": datetime.now().astimezone().isoformat(),
        "generator": "generate_datasets.py",
        "seed": SEED,
        "references": [
            "3GPP TS 38.300, 38.331, 38.215, 38.214, 28.530, 28.105, 23.288, 33.501",
            "3GPP TR 38.817, TR 23.700-80",
            "O-RAN Near-RT RIC, E2GAP, xApp/rApp specifications",
            "NBUC Project Problem Statement - Swetha Kerahalli",
        ],
        "rag_corpus": {
            "file": rag_path.name,
            "num_chunks": len(corpus),
            "categories": sorted({c["category"] for c in corpus}),
            "format": "jsonl",
            "fields": ["doc_id", "chunk_id", "title", "category", "source", "content", "keywords", "metadata"],
        },
        "telemetry_csv": csv_meta,
    }
    meta_path = DATA_DIR / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"RAG corpus: {rag_path} ({len(corpus)} chunks)")
    print(f"Telemetry CSV: {csv_path} ({NUM_ROWS} rows x {len(CSV_COLUMNS)} columns)")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
