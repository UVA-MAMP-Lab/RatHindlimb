from pathlib import Path

# Model and setup file paths
models_path = Path("../../models/osim/")
unscaled_model_path = models_path / "rat_hindlimb_bilateral.osim"
marker_set_path = models_path / "rat_hindlimb_bilateral_markerset.xml"
generic_scale_setup_path = models_path / "rat_hindlimb_bilateral_scale_setup.xml"
generic_ik_setup_path = models_path / "rat_hindlimb_bilateral_ik_setup.xml"

# Define required markers and parameters for validation
required_markers = [
    "TAIL",
    "SPL6",
    "LASI",
    "RASI",  # Torso
    "LHIP",
    "LKNE",
    "LANK",
    "LTOE",  # Left leg
    "RHIP",
    "RKNE",
    "RANK",
    "RTOE",  # Right leg
]

required_parameters = [
    "Mass",
    "Length",
    "RFemurLength",
    "RTibiaLength",
    "RFootLength",
    "LFemurLength",
    "LTibiaLength",
    "LFootLength",
]


def process_session():
    pass
