from dataclasses import dataclass
from pathlib import Path
import opensim as osim
from osimpy import IKSettings, IDSettings, CMCSettings

# Model and setup file paths
models_dir = Path("../../models/osim/")
unscaled_model_path = models_dir / "rat_hindlimb_bilateral.osim"
marker_set_path = models_dir / "rat_hindlimb_bilateral_markerset.xml"
generic_scale_setup_path = models_dir / "rat_hindlimb_bilateral_scale_setup.xml"
generic_ik_setup_path = models_dir / "rat_hindlimb_bilateral_ik_setup.xml"

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


@dataclass
class ModelBranchPaths:
    """Branch-aware paths for model artifacts.

    Intermediates are written to ``model_dir/.build/<branch>/`` so the top-level
    models directory remains uncluttered.
    """

    model_dir: str | Path
    branch: str = "with_muscles"
    keep_intermediates: bool = False

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.intermediate_dir = self.model_dir / ".build" / self.branch
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, stage: str, suffix: str = ".osim") -> Path:
        stage_slug = stage.strip().replace(" ", "_").replace("/", "_")
        return self.intermediate_dir / f"{stage_slug}{suffix}"

    def temp_path(self, filename: str) -> Path:
        return self.intermediate_dir / filename

    def final_path(self, model_name: str) -> Path:
        if model_name.endswith(".osim"):
            return self.model_dir / model_name
        return self.model_dir / f"{model_name}.osim"

    def cleanup(self):
        if self.keep_intermediates or not self.intermediate_dir.exists():
            return
        for path in self.intermediate_dir.glob("*"):
            if path.is_file():
                path.unlink()


def remove_muscles(model: osim.Model) -> osim.Model:
    """Remove all muscles from a model in-place and return the same model."""
    force_set: osim.ForceSet = model.upd_ForceSet()
    indices_to_remove = []
    for i in range(force_set.getSize()):
        if osim.Muscle.safeDownCast(force_set.get(i)) is not None:
            indices_to_remove.append(i)

    for i in indices_to_remove[::-1]:
        force_set.remove(i)

    return model


def update_model(
    model: osim.Model,
    save_path: str | Path | None = None,
    *,
    branch_paths: ModelBranchPaths | None = None,
    stage: str | None = None,
    final_name: str | None = None,
) -> osim.Model:
    """
    Helper function to update and save the OpenSim model.

    Returns the updated model.
    """
    if save_path is None:
        if branch_paths is not None and stage is not None:
            save_path = branch_paths.checkpoint_path(stage)
        elif branch_paths is not None and final_name is not None:
            save_path = branch_paths.final_path(final_name)
        else:
            raise ValueError(
                "Provide save_path, or provide branch_paths with either stage or final_name."
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.finalizeFromProperties()
    model.finalizeConnections()
    model.printToXML(str(save_path))

    return osim.Model(str(save_path))

# Convenient wrappers for running OpenSim tools for the rat hindlimb model
def run_ik():

    pass

def run_id():
    pass

def run_moco_inverse():
    pass
