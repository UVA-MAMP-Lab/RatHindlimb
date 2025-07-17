import os
import glob
from src.processing import process_session  

root_dir = 'data'

# Define the control group specification (This will eventually be a SQL query)
# For example, this would be SELECT * FROM sessions WHERE classification IN ('AFIRM', 'BAA') AND session_name = 'Baseline'
control_spec = {
    # "AFIRM": {
    #     "*": [  # All subjects
    #         "Baseline"  # All trials in Baseline sessions
    #     ]
    # },
    "BAA": {
        "*": [ # All subjects
            "Baseline" # All trials in Baseline sessions
        ] 
    }
}

def create_session_globs(spec: dict) -> list[str]:
    """
    Create glob patterns for session directories based on the control group specification.
    """
    session_globs = []
    for classification, subjects in spec.items():
        for subject_pattern, sessions in subjects.items():
            for session in sessions:
                if subject_pattern == "*":
                    # Use glob wildcard for all subjects
                    pattern = os.path.join(classification, "*", session)
                else:
                    # Use specific subject name
                    pattern = os.path.join(classification, subject_pattern, session)
                session_globs.append(pattern)
                print(f"Created pattern: {pattern}")  # Debug output
    return session_globs

results = {} # Should be structured as {<classification>: {<subject>: {<session>: <session_results>}}
for session_glob in create_session_globs(control_spec):
    # Find all matching session directories
    session_dirs = glob.glob(os.path.join(root_dir, session_glob))

    for session_dir in session_dirs:
        # Process each session directory
        print(f"Processing session: {session_dir}")
        try:
            session_results = process_session(session_dir)
            classification = os.path.basename(os.path.dirname(os.path.dirname(session_dir)))
            subject = os.path.basename(os.path.dirname(session_dir))
            session_name = os.path.basename(session_dir)

            if classification not in results:
                results[classification] = {}
            if subject not in results[classification]:
                results[classification][subject] = {}
            results[classification][subject][session_name] = session_results

        except Exception as e:
            print(f"Error processing session {session_dir}: {e}")

import json
# Save results to a JSON file
output_file = os.path.join(root_dir, 'control_group_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Control group results saved to {output_file}")