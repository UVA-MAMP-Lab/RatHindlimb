from myoconverter.O2MPipeline import O2MPipeline

kwargs = {}
# General configure
kwargs = {}  # define kwargs inputs
kwargs['convert_steps'] = [1, 2]                # All three steps selected
kwargs['muscle_list'] = None                    # No specific muscle selected, optimize all of them
kwargs['osim_data_overwrite'] = True            # Overwrite the Osim model state files
kwargs['conversion'] = True                     # Yes, perform 'Cvt#' process
kwargs['validation'] = True                     # Yes, perform 'Vlt#' process
kwargs['generate_pdf'] = False                  # Do not generate validation pdf report
kwargs['speedy'] = False                        # Do not reduce the checking notes to increase speed
kwargs['add_ground_geom'] = True                # Add ground to the model
kwargs['treat_as_normal_path_point'] = False    # Use original constraints to represent moving and conditional path points

# Osim model info & target saving folder
osim_file = '../models/rat_hindlimb_bilateral.osim'
geometry_folder = '../models/Geometry'
output_folder = '../models/converted'

# Run pipeline
O2MPipeline(osim_file, geometry_folder, output_folder, **kwargs)