from pathlib import Path
import opensim as osim


def update_model(
    model: osim.Model,
    save_path: str | Path,
) -> osim.Model:
    """
    Helper function to update and save the OpenSim model.

    Returns the updated model.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.finalizeFromProperties()
    model.finalizeConnections()
    model.printToXML(str(save_path))

    return osim.Model(str(save_path))
