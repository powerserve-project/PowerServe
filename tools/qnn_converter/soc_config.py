from dataclasses import dataclass


@dataclass
class SoCConfig:
    htp_version: int
    soc_id: int


soc_map = {
    "8650": SoCConfig(htp_version=75, soc_id=57),
    "8750": SoCConfig(htp_version=79, soc_id=69),
    "8295": SoCConfig(htp_version=68, soc_id=39),
}
