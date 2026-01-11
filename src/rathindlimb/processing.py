from pathlib import Path
import opensim as osim

#
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


def update_model(model: osim.Model, save_path: str | Path) -> osim.Model:
    """
    Helper function to update and save the OpenSim model.

    Returns the updated model.
    """
    model.finalizeFromProperties()
    model.finalizeConnections()
    model.printToXML(str(save_path))

    return osim.Model(str(save_path))


def run_moco_inverse():
    pass
