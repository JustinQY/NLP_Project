########################################################
# remove beginning lines and collect the actors' lines
########################################################

import re

def clean_script(script):
    lines = script.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("Written by") or line.startswith("[Scene"):
            continue
        if ':' in line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)