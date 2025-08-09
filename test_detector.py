import os
import cv2
import numpy as np
from app import EmergencyLightingDetector

def test_emergency_detector():
    # Initialize the detector with the updated symbols
    detector = EmergencyLightingDetector()
    
    # Print the emergency symbols and fixture types
    print("Emergency Symbols:")
    print(detector.emergency_symbols)
    print("\nFixture Types:")
    for symbol, description in detector.fixture_types.items():
        print(f"  {symbol}: {description}")
    
    # Test the detector on some sample text
    test_texts = [
        ["A1E", "Exit/Emergency", "Combo", "Unit"],
        ["A1-E", "Type", "A1", "Emergency", "Fixture"],
        ["A1/E", "Type", "A1", "with", "Emergency", "Battery", "Backup"],
        ["EM-1", "Emergency", "Type", "1", "Fixture"],
        ["EM-2", "Emergency", "Type", "2", "Fixture"],
        ["EXIT-EM", "Exit", "Sign", "with", "Emergency", "Backup"],
        ["EL", "Emergency", "Light"],
        ["Regular", "lighting", "fixture", "not", "emergency"],
        ["Standard", "2x4", "fixture", "no", "emergency", "backup"]
    ]
    
    # Debug the problematic cases
    print("\nSpecial debugging for A1-E and A1/E cases:")
    for case in [["A1-E", "Type", "A1", "Emergency", "Fixture"], ["A1/E", "Type", "A1", "with", "Emergency", "Battery", "Backup"]]:
        print(f"\nCase: {' '.join(case)}")
        print(f"  First word: {case[0]}")
        print(f"  First word upper: {case[0].upper()}")
        print(f"  Check if first word equals 'A1-E': {case[0].upper() == 'A1-E'}")
        print(f"  Check if first word equals 'A1/E': {case[0].upper() == 'A1/E'}")
        print(f"  ASCII values of first word: {[ord(c) for c in case[0]]}")
        print(f"  ASCII values of 'A1-E': {[ord(c) for c in 'A1-E']}")
        print(f"  ASCII values of 'A1/E': {[ord(c) for c in 'A1/E']}")
        print(f"  Text string: {' '.join(case)}")
        print(f"  Text string upper: {' '.join(case).upper()}")
        print(f"  'A1-E' in text_str.upper().split(): {'A1-E' in ' '.join(case).upper().split()}")
        print(f"  'A1/E' in text_str.upper().split(): {'A1/E' in ' '.join(case).upper().split()}")
    
    # Add some additional test cases for the specific A1-E and A1/E patterns
    additional_tests = [
        ["This", "is", "an", "A1-E", "emergency", "fixture"],
        ["This", "is", "an", "A1/E", "emergency", "fixture"]
    ]
    test_texts.extend(additional_tests)
    
    # Print the raw text for debugging
    print("\nRaw text for each test case:")
    for words in test_texts:
        print(f"  {' '.join(words)}")
        print(f"  Words: {words}")
    
    print("\nTesting emergency fixture detection:")
    for words in test_texts:
        # Convert to string for words in test_texts:
        text_str = ' '.join(words)
        is_emergency = detector.is_emergency_fixture(words)
        
        # Debug info for A1-E and A1/E cases
        if words[0] in ['A1-E', 'A1/E'] or any(pattern in words for pattern in ['A1-E', 'A1/E']):
            print(f"DEBUG for A1-E/A1/E pattern:")
            print(f"  Words: {words}")
            print(f"  First word: {words[0]}")
            print(f"  First word upper: {words[0].upper()}")
            print(f"  Check if first word equals 'A1-E': {words[0].upper() == 'A1-E'}")
            print(f"  Check if first word equals 'A1/E': {words[0].upper() == 'A1/E'}")
            print(f"  Text string: {text_str}")
            print(f"  Text string upper: {text_str.upper()}")
            print(f"  'A1-E' in text_str.upper().split(): {'A1-E' in text_str.upper().split()}")
            print(f"  'A1/E' in text_str.upper().split(): {'A1/E' in text_str.upper().split()}")
        
        # Test identify_symbol method
        symbol = detector.identify_symbol(words) if is_emergency else "None"
        description = detector.fixture_types.get(symbol, f"Unknown Fixture Type ({symbol})") if symbol != "None" else "Not an emergency fixture"
        
        print(f"Text: '{text_str}'")
        print(f"  Is Emergency: {is_emergency}")
        print(f"  Symbol: {symbol}")
        print(f"  Description: {description}")
        print()

if __name__ == "__main__":
    test_emergency_detector()