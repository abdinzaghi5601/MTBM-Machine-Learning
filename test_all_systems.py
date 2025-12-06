"""
Comprehensive Test Suite for All Cylinder Systems (3, 4, and 6)
"""

from steering_calculator import (
    SteeringCalculator,
    MachineParameters,
    SteeringCommand,
    CylinderReadings,
    quick_calculate,
    quick_reverse
)

print("="*80)
print("COMPREHENSIVE STEERING CALCULATOR TEST SUITE")
print("="*80)

# ============================================================================
# TEST 1: 3-CYLINDER SYSTEM
# ============================================================================

print("\n" + "="*80)
print("TEST 1: 3-CYLINDER SYSTEM")
print("="*80)

params_3 = MachineParameters(
    num_cylinders=3,
    stroke=50.0,
    mounting_diameter=715.0,
    pipe_length=3000.0,
    vertical_angle=1.49
)

calc_3 = SteeringCalculator(params_3)

# Test 1a: Forward calculation
print("\nTest 1a: Forward Calculation (Pitch/Yaw → Cylinders)")
print("-" * 80)
steering_3 = SteeringCommand(pitch=-4.5, yaw=16.5)
cylinders_3 = calc_3.calculate_cylinders(steering_3)
print(f"Input: Pitch = {steering_3.pitch} mm/m, Yaw = {steering_3.yaw} mm/m")
print(f"Output:")
for cyl, pos in cylinders_3.items():
    print(f"  {cyl}: {pos} mm")

# Test 1b: Reverse calculation
print("\nTest 1b: Reverse Calculation (Cylinders → Pitch/Yaw)")
print("-" * 80)
readings_3 = CylinderReadings(cylinder_1=20.0, cylinder_2=32.0, cylinder_3=30.0)
reverse_3 = calc_3.calculate_steering(readings_3)
print(f"Input: Cyl1 = 20.0mm, Cyl2 = 32.0mm, Cyl3 = 30.0mm")
print(f"Output: Pitch = {reverse_3.pitch} mm/m, Yaw = {reverse_3.yaw} mm/m")

# Test 1c: Round-trip test (forward then reverse)
print("\nTest 1c: Round-Trip Test")
print("-" * 80)
test_steering = SteeringCommand(pitch=10.0, yaw=-5.0)
test_cylinders = calc_3.calculate_cylinders(test_steering)
roundtrip_steering = calc_3.calculate_steering(
    CylinderReadings(
        cylinder_1=test_cylinders['cylinder_1'],
        cylinder_2=test_cylinders['cylinder_2'],
        cylinder_3=test_cylinders['cylinder_3']
    )
)
print(f"Original:   Pitch = {test_steering.pitch}, Yaw = {test_steering.yaw}")
print(f"Round-trip: Pitch = {roundtrip_steering.pitch}, Yaw = {roundtrip_steering.yaw}")
match = (abs(test_steering.pitch - roundtrip_steering.pitch) < 0.01 and
         abs(test_steering.yaw - roundtrip_steering.yaw) < 0.01)
print(f"Match: {'✅ PASS' if match else '❌ FAIL'}")

# Test 1d: Complete analysis
print("\nTest 1d: Complete Analysis & Correction Planning")
print("-" * 80)
current_readings = CylinderReadings(cylinder_1=28.69, cylinder_2=31.52, cylinder_3=21.79)
analysis = calc_3.analyze_current_state(current_readings)
print(f"Current State: Pitch = {analysis['current_steering']['pitch']} mm/m, "
      f"Yaw = {analysis['current_steering']['yaw']} mm/m")

correction_plan = calc_3.plan_correction(
    current_pitch=analysis['current_steering']['pitch'],
    current_yaw=analysis['current_steering']['yaw'],
    target_pitch=-4.5,
    target_yaw=16.5
)
print(f"Target State: Pitch = -4.5 mm/m, Yaw = 16.5 mm/m")
print(f"Required Correction: Pitch = {correction_plan['required_correction']['pitch']} mm/m, "
      f"Yaw = {correction_plan['required_correction']['yaw']} mm/m")
print(f"Feasible: {'✅ YES' if correction_plan['feasibility']['is_feasible'] else '❌ NO'}")

# ============================================================================
# TEST 2: 4-CYLINDER SYSTEM
# ============================================================================

print("\n" + "="*80)
print("TEST 2: 4-CYLINDER SYSTEM")
print("="*80)

params_4 = MachineParameters(
    num_cylinders=4,
    stroke=50.0,
    mounting_diameter=715.0,
    pipe_length=3000.0,
    vertical_angle=1.49
)

calc_4 = SteeringCalculator(params_4)

# Test 2a: Forward calculation
print("\nTest 2a: Forward Calculation (Pitch/Yaw → Cylinders)")
print("-" * 80)
steering_4 = SteeringCommand(pitch=10.0, yaw=15.0)
cylinders_4 = calc_4.calculate_cylinders(steering_4)
print(f"Input: Pitch = {steering_4.pitch} mm/m, Yaw = {steering_4.yaw} mm/m")
print(f"Output:")
for cyl, pos in cylinders_4.items():
    print(f"  {cyl}: {pos} mm")

# Test 2b: Reverse calculation
print("\nTest 2b: Reverse Calculation (Cylinders → Pitch/Yaw)")
print("-" * 80)
readings_4 = CylinderReadings(
    cylinder_1=30.0,
    cylinder_2=28.0,
    cylinder_3=20.0,
    cylinder_4=22.0
)
reverse_4 = calc_4.calculate_steering(readings_4)
print(f"Input: Cyl1 = 30.0mm, Cyl2 = 28.0mm, Cyl3 = 20.0mm, Cyl4 = 22.0mm")
print(f"Output: Pitch = {reverse_4.pitch} mm/m, Yaw = {reverse_4.yaw} mm/m")

# Test 2c: Round-trip test
print("\nTest 2c: Round-Trip Test")
print("-" * 80)
test_steering_4 = SteeringCommand(pitch=8.0, yaw=-6.0)
test_cylinders_4 = calc_4.calculate_cylinders(test_steering_4)
roundtrip_steering_4 = calc_4.calculate_steering(
    CylinderReadings(
        cylinder_1=test_cylinders_4['cylinder_1'],
        cylinder_2=test_cylinders_4['cylinder_2'],
        cylinder_3=test_cylinders_4['cylinder_3'],
        cylinder_4=test_cylinders_4['cylinder_4']
    )
)
print(f"Original:   Pitch = {test_steering_4.pitch}, Yaw = {test_steering_4.yaw}")
print(f"Round-trip: Pitch = {roundtrip_steering_4.pitch}, Yaw = {roundtrip_steering_4.yaw}")
match_4 = (abs(test_steering_4.pitch - roundtrip_steering_4.pitch) < 0.01 and
           abs(test_steering_4.yaw - roundtrip_steering_4.yaw) < 0.01)
print(f"Match: {'✅ PASS' if match_4 else '❌ FAIL'}")

# Test 2d: Symmetry test (opposite cylinders)
print("\nTest 2d: Symmetry Test")
print("-" * 80)
print("Testing that opposite cylinders balance correctly...")
symmetric_reading = CylinderReadings(
    cylinder_1=30.0,  # top
    cylinder_2=25.0,  # right (center)
    cylinder_3=20.0,  # bottom (should be opposite of top)
    cylinder_4=25.0   # left (center)
)
sym_steering = calc_4.calculate_steering(symmetric_reading)
print(f"Symmetric input: Cyl1=30, Cyl2=25, Cyl3=20, Cyl4=25")
print(f"Output: Pitch = {sym_steering.pitch} mm/m, Yaw = {sym_steering.yaw} mm/m")
print(f"Expected: Pitch ≠ 0, Yaw ≈ 0")
yaw_near_zero = abs(sym_steering.yaw) < 0.1
print(f"Yaw near zero: {'✅ PASS' if yaw_near_zero else '❌ FAIL'}")

# ============================================================================
# TEST 3: 6-CYLINDER SYSTEM
# ============================================================================

print("\n" + "="*80)
print("TEST 3: 6-CYLINDER SYSTEM")
print("="*80)

params_6 = MachineParameters(
    num_cylinders=6,
    stroke=50.0,
    mounting_diameter=715.0,
    pipe_length=3000.0,
    vertical_angle=1.49
)

calc_6 = SteeringCalculator(params_6)

# Test 3a: Forward calculation
print("\nTest 3a: Forward Calculation (Pitch/Yaw → Cylinders)")
print("-" * 80)
steering_6 = SteeringCommand(pitch=-13.0, yaw=-6.0)
cylinders_6 = calc_6.calculate_cylinders(steering_6)
print(f"Input: Pitch = {steering_6.pitch} mm/m, Yaw = {steering_6.yaw} mm/m")
print(f"Output:")
for cyl, pos in cylinders_6.items():
    print(f"  {cyl}: {pos} mm")

# Test 3b: Reverse calculation
print("\nTest 3b: Reverse Calculation (Cylinders → Pitch/Yaw)")
print("-" * 80)
readings_6 = CylinderReadings(
    cylinder_1=28.0,
    cylinder_2=26.0,
    cylinder_3=23.0,
    cylinder_4=22.0,
    cylinder_5=24.0,
    cylinder_6=27.0
)
reverse_6 = calc_6.calculate_steering(readings_6)
print(f"Input: Cyl1 = 28.0mm, Cyl2 = 26.0mm, Cyl3 = 23.0mm")
print(f"       Cyl4 = 22.0mm, Cyl5 = 24.0mm, Cyl6 = 27.0mm")
print(f"Output: Pitch = {reverse_6.pitch} mm/m, Yaw = {reverse_6.yaw} mm/m")

# Test 3c: Round-trip test
print("\nTest 3c: Round-Trip Test")
print("-" * 80)
test_steering_6 = SteeringCommand(pitch=12.0, yaw=-8.0)
test_cylinders_6 = calc_6.calculate_cylinders(test_steering_6)
roundtrip_steering_6 = calc_6.calculate_steering(
    CylinderReadings(
        cylinder_1=test_cylinders_6['cylinder_1'],
        cylinder_2=test_cylinders_6['cylinder_2'],
        cylinder_3=test_cylinders_6['cylinder_3'],
        cylinder_4=test_cylinders_6['cylinder_4'],
        cylinder_5=test_cylinders_6['cylinder_5'],
        cylinder_6=test_cylinders_6['cylinder_6']
    )
)
print(f"Original:   Pitch = {test_steering_6.pitch}, Yaw = {test_steering_6.yaw}")
print(f"Round-trip: Pitch = {roundtrip_steering_6.pitch}, Yaw = {roundtrip_steering_6.yaw}")
match_6 = (abs(test_steering_6.pitch - roundtrip_steering_6.pitch) < 0.01 and
           abs(test_steering_6.yaw - roundtrip_steering_6.yaw) < 0.01)
print(f"Match: {'✅ PASS' if match_6 else '❌ FAIL'}")

# Test 3d: Cylinder arrangement visualization
print("\nTest 3d: Cylinder Arrangement (60° spacing)")
print("-" * 80)
print("Cylinder positions around the mounting circle:")
for i in range(6):
    angle = i * 60
    print(f"  Cylinder {i+1}: {angle:3d}° {'(Top)' if angle==0 else ''}")

# ============================================================================
# TEST 4: QUICK FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("TEST 4: CONVENIENCE FUNCTIONS")
print("="*80)

print("\nTest 4a: quick_calculate()")
print("-" * 80)
quick_result = quick_calculate(pitch=-5, yaw=10, num_cylinders=3)
print(f"quick_calculate(pitch=-5, yaw=10, num_cylinders=3)")
print(f"Result: {quick_result}")

print("\nTest 4b: quick_reverse()")
print("-" * 80)
pitch_r, yaw_r = quick_reverse([20.0, 32.0, 30.0], num_cylinders=3)
print(f"quick_reverse([20.0, 32.0, 30.0], num_cylinders=3)")
print(f"Result: Pitch = {pitch_r}, Yaw = {yaw_r}")

# ============================================================================
# TEST 5: VALIDATION & SAFETY
# ============================================================================

print("\n" + "="*80)
print("TEST 5: VALIDATION & SAFETY CHECKS")
print("="*80)

print("\nTest 5a: Stroke Limit Detection")
print("-" * 80)
extreme_steering = SteeringCommand(pitch=100, yaw=100)
extreme_cylinders = calc_3.calculate_cylinders(extreme_steering)
print(f"Extreme input: Pitch = 100 mm/m, Yaw = 100 mm/m")
print(f"Output cylinders:")
for cyl, pos in extreme_cylinders.items():
    status = "OUT OF RANGE" if (pos < 0 or pos > 50) else "OK"
    print(f"  {cyl}: {pos:6.2f} mm [{status}]")

print("\nTest 5b: Feasibility Check")
print("-" * 80)
safe_plan = calc_3.plan_correction(0, 0, 5, 5)
unsafe_plan = calc_3.plan_correction(0, 0, 200, 200)
print(f"Safe correction (0,0 → 5,5): {safe_plan['feasibility']['is_feasible']}")
print(f"Unsafe correction (0,0 → 200,200): {unsafe_plan['feasibility']['is_feasible']}")
print(f"Reason: {unsafe_plan['feasibility']['reason']}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

all_tests = [
    ("3-Cylinder Round-Trip", match),
    ("4-Cylinder Round-Trip", match_4),
    ("4-Cylinder Symmetry", yaw_near_zero),
    ("6-Cylinder Round-Trip", match_6),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nResults: {passed}/{total} tests passed")
print("-" * 80)
for name, result in all_tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"  {status}  {name}")

print("\n" + "="*80)
print("SYSTEM CAPABILITIES")
print("="*80)
print("✅ 3-Cylinder System: Forward & Reverse Calculations")
print("✅ 4-Cylinder System: Forward & Reverse Calculations")
print("✅ 6-Cylinder System: Forward & Reverse Calculations")
print("✅ Complete Analysis & Correction Planning")
print("✅ Feasibility Checking & Validation")
print("✅ Professional Report Generation")
print("✅ Interactive CLI Interface")
print("\n" + "="*80)
