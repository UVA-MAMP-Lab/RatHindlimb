import os
import json
from src.processing import process_spec  


root_dir = os.path.join('C:\\', 'Users', 'hpb7kr', 'OneDrive - University of Virginia', 'Shared Documents - MAMP Lab Folder', 'General', 'MoCapData', 'ViconData', 'Rats')

# Define the control group specification (This will eventually be a SQL query)
# # For example, this would be SELECT * FROM sessions WHERE classification IN ('AFIRM', 'BAA') AND session_name = 'Baseline'
# control_spec = {
#     "AFIRM": {
#         "*": [  # All subjects except Old01-Old03
#             "Baseline"  # All trials in Baseline sessions
#         ]
#     },
#     "BAA": {
#         "*": [ # All subjects
#             "Baseline" # All trials in Baseline sessions
#         ] 
#     }
# }

# results = process_spec(root_dir, control_spec)

# # Save results to a JSON file
# output_file = os.path.join(root_dir, 'control_group_results.json')
# with open(output_file, 'w') as f:
#     json.dump(results, f, indent=4)
# print(f"Control group results saved to {output_file}")

old_spec = {
    "AFIRM": {
        "Old01": [
            "Baseline"  # All trials in Baseline sessions
        ],
        "Old02": [
            "Baseline"  # All trials in Baseline sessions
        ],
        "Old03": [
            "Baseline"  # All trials in Baseline sessions
        ]
    }
}

results_old = process_spec(root_dir, old_spec)
output_file_old = os.path.join(root_dir, 'old_group_results.json')
with open(output_file_old, 'w') as f:
    json.dump(results_old, f, indent=4)
print(f"Old group results saved to {output_file_old}")
