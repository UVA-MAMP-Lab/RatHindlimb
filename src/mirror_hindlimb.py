import opensim as osim

# Define naming schemes for original and mirrored objects
original_bodies = f"_r"
mirrored_bodies = f"_l"

original_wrap = f"_r"
mirrored_wrap = f"_l"

original_markers = f"R"
mirrored_markers = f"L"

# Define body and plane for mirroring
mirror_body = 'spine' # body is optional: will mirror all bodies across the origin if not specified
mirror_plane = 'xy'

# Load the unilateral model
model = osim.Model('rat_hindlimb.osim')
# Clone the model to create a new mirrored model
new_model = model.clone()

# Mirror across some plane

## For each body to mirrr
# Mirror geometry

# Mirror wrap objects
# rename
# Negate x and y rotation 
# Negate z translation


## For each joint to mirror

## For each marker to mirror
# Update original parent frame
# Create new marker with mirrored name and location





