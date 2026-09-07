"""Single source of truth for CICIoT2023 label -> 8-class mapping.

The raw dataset ships 34 fine-grained labels (33 attacks + BenignTraffic).
This project's target is the 8-class scheme locked in CLAUDE.md: Benign plus
the 7 attack categories (DDoS, DoS, Recon, Web-based, Brute Force, Spoofing,
Mirai). Do not hardcode category strings anywhere else — import from here.
"""

from __future__ import annotations

BENIGN = "Benign"
DDOS = "DDoS"
DOS = "DoS"
RECON = "Recon"
WEB_BASED = "Web-based"
BRUTE_FORCE = "Brute Force"
SPOOFING = "Spoofing"
MIRAI = "Mirai"

CLASSES = [BENIGN, DDOS, DOS, RECON, WEB_BASED, BRUTE_FORCE, SPOOFING, MIRAI]

# Raw CICIoT2023 label -> 8-class category.
# Raw labels as they appear in the official/Kaggle-mirror CSVs.
LABEL_TO_CLASS: dict[str, str] = {
    "BenignTraffic": BENIGN,
    # DDoS (12 subtypes)
    "DDoS-RSTFINFlood": DDOS,
    "DDoS-PSHACK_Flood": DDOS,
    "DDoS-SYN_Flood": DDOS,
    "DDoS-UDP_Flood": DDOS,
    "DDoS-TCP_Flood": DDOS,
    "DDoS-ICMP_Flood": DDOS,
    "DDoS-SynonymousIP_Flood": DDOS,
    "DDoS-ACK_Fragmentation": DDOS,
    "DDoS-UDP_Fragmentation": DDOS,
    "DDoS-ICMP_Fragmentation": DDOS,
    "DDoS-SlowLoris": DDOS,
    "DDoS-HTTP_Flood": DDOS,
    # DoS (4 subtypes)
    "DoS-UDP_Flood": DOS,
    "DoS-SYN_Flood": DOS,
    "DoS-TCP_Flood": DOS,
    "DoS-HTTP_Flood": DOS,
    # Recon (5 subtypes)
    "Recon-PingSweep": RECON,
    "Recon-OSScan": RECON,
    "Recon-PortScan": RECON,
    "VulnerabilityScan": RECON,
    "Recon-HostDiscovery": RECON,
    # Web-based (6 subtypes)
    "BrowserHijacking": WEB_BASED,
    "Backdoor_Malware": WEB_BASED,
    "XSS": WEB_BASED,
    "Uploading_Attack": WEB_BASED,
    "SqlInjection": WEB_BASED,
    "CommandInjection": WEB_BASED,
    # Brute Force (1 subtype)
    "DictionaryBruteForce": BRUTE_FORCE,
    # Spoofing (2 subtypes)
    "DNS_Spoofing": SPOOFING,
    "MITM-ArpSpoofing": SPOOFING,
    # Mirai (3 subtypes)
    "Mirai-greeth_flood": MIRAI,
    "Mirai-greip_flood": MIRAI,
    "Mirai-udpplain": MIRAI,
}

assert len(LABEL_TO_CLASS) == 34, "expected 33 attacks + Benign = 34 raw labels"


def to_class(raw_label: str) -> str:
    """Map a raw CICIoT2023 label to its 8-class category.

    Raises KeyError on an unrecognized label rather than silently bucketing
    it — an unmapped label usually means the raw file's label spelling
    differs from what's captured here (verify against the source CSVs
    before adding new keys, per CLAUDE.md's Kaggle-mirror caveat).
    """
    return LABEL_TO_CLASS[raw_label]
