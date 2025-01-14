from dataclasses import dataclass


@dataclass
class SoCConfig:
    htp_version: int
    soc_id: int


soc_map = {
    "8gen3": SoCConfig(htp_version=75, soc_id=57),
    "8gen4": SoCConfig(htp_version=79, soc_id=69),
}
