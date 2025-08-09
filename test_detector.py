import os
import json
import re

def update_emergency_detector():
    # Read the current app.py file
    app_py_path = 'app.py'
    if not os.path.exists(app_py_path):
        print(f"Error: {app_py_path} not found")
        return
    
    with open(app_py_path, 'r') as f:
        app_py_content = f.read()
    
    # Find the EmergencyLightingDetector class initialization
    detector_init_start = app_py_content.find('class EmergencyLightingDetector:')
    if detector_init_start == -1:
        print("Error: Could not find EmergencyLightingDetector class in app.py")
        return
    
    # Find the emergency_symbols and fixture_types definitions
    symbols_start = app_py_content.find('self.emergency_symbols = [', detector_init_start)
    symbols_end = app_py_content.find(']', symbols_start) + 1
    
    fixture_types_start = app_py_content.find('self.fixture_types = {', detector_init_start)
    fixture_types_end = app_py_content.find('}', fixture_types_start) + 1
    
    if symbols_start == -1 or fixture_types_start == -1:
        print("Error: Could not find emergency_symbols or fixture_types in app.py")
        return
    
    # Extract current symbols and fixture types
    current_symbols_str = app_py_content[symbols_start:symbols_end]
    current_fixture_types_str = app_py_content[fixture_types_start:fixture_types_end]
    
    print("\nCurrent emergency symbols and fixture types:")
    print(current_symbols_str)
    print(current_fixture_types_str)
    
    # Define new symbols and fixture types based on PDF analysis
    # These are examples based on common emergency lighting symbols found in the PDF
    new_symbols = ['A1-E', 'A1/E', 'EM-1', 'EM-2', 'EXIT-EM', 'EL']
    new_fixture_types = {
        'A1-E': 'Type A1 Emergency Fixture',
        'A1/E': 'Type A1 with Emergency Battery Backup',
        'EM-1': 'Emergency Type 1 Fixture',
        'EM-2': 'Emergency Type 2 Fixture',
        'EXIT-EM': 'Exit Sign with Emergency Backup',
        'EL': 'Emergency Light'
    }
    
    # Extract current symbols as a list
    current_symbols_match = re.search(r"self\.emergency_symbols = \[([^\]]*)\]", current_symbols_str)
    if current_symbols_match:
        symbols_content = current_symbols_match.group(1)
        current_symbols = re.findall(r"'([^']*)'|\"([^\"]*)\"|", symbols_content)
        current_symbols = [s[0] or s[1] for s in current_symbols if s[0] or s[1]]
    else:
        current_symbols = []
    
    # Extract current fixture types as a dictionary
    current_fixture_types = {}
    fixture_type_matches = re.findall(r"'([^']*)': '([^']*)'|\"([^\"]*)\":\s*\"([^\"]*)\"|", current_fixture_types_str)
    for match in fixture_type_matches:
        if match[0] and match[1]:
            current_fixture_types[match[0]] = match[1]
        elif match[2] and match[3]:
            current_fixture_types[match[2]] = match[3]
    
    # Add new symbols that don't already exist
    updated_symbols = current_symbols.copy()
    for symbol in new_symbols:
        if symbol not in updated_symbols:
            updated_symbols.append(symbol)
    
    # Add new fixture types that don't already exist
    updated_fixture_types = current_fixture_types.copy()
    for symbol, description in new_fixture_types.items():
        if symbol not in updated_fixture_types:
            updated_fixture_types[symbol] = description
    
    # Format updated symbols as a string
    updated_symbols_str = "self.emergency_symbols = [" + ", ".join([f"'{s}'" for s in updated_symbols]) + "]"
    
    # Format updated fixture types as a string
    updated_fixture_types_str = "self.fixture_types = {\n"
    for symbol, description in updated_fixture_types.items():
        updated_fixture_types_str += f"            '{symbol}': '{description}',\n"
    updated_fixture_types_str = updated_fixture_types_str.rstrip(",\n") + "\n        }"
    
    # Replace the old definitions with the updated ones
    # First, create a copy of the original content
    updated_app_py_content = app_py_content
    
    # Replace the fixture_types first (since it appears later in the file)
    updated_app_py_content = updated_app_py_content[:fixture_types_start] + updated_fixture_types_str + updated_app_py_content[fixture_types_end:]
    
    # Then replace the emergency_symbols
    updated_app_py_content = updated_app_py_content[:symbols_start] + updated_symbols_str + updated_app_py_content[symbols_end:]
    
    # Write the updated content back to app.py
    with open('app.py.updated', 'w') as f:
        f.write(updated_app_py_content)
    
    print("\nUpdated emergency symbols and fixture types:")
    print(updated_symbols_str)
    print(updated_fixture_types_str)
    print("\nUpdated app.py saved as app.py.updated")
    print("Review the changes and rename the file to app.py if they look good.")

if __name__ == "__main__":
    update_emergency_detector()
