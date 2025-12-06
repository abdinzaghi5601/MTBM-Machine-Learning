"""
Interactive Command-Line Interface for Steering Calculator
===========================================================

User-friendly interactive interface for microtunneling steering calculations.
Provides step-by-step prompts for entering machine parameters, cylinder readings,
and target steering states, then generates comprehensive reports.

Usage:
    python3 steering_cli.py

Author: Reverse-engineered from Steer-cyl-cal-rev8_.xls (S.J.Baba)
Version: 2.0
"""

from steering_calculator import (
    MachineParameters,
    SteeringCommand,
    CylinderReadings,
    SteeringCalculator
)


def get_float_input(prompt: str, default: float) -> float:
    """
    Get float input from user with default value

    Args:
        prompt: Prompt message to display
        default: Default value if user presses Enter

    Returns:
        Float value from user or default
    """
    while True:
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            return float(value) if value else default
        except ValueError:
            print("  Invalid input. Please enter a number.")


def get_int_input(prompt: str, default: int, valid_values: list = None) -> int:
    """
    Get integer input from user with default value

    Args:
        prompt: Prompt message to display
        default: Default value if user presses Enter
        valid_values: Optional list of valid values

    Returns:
        Integer value from user or default
    """
    while True:
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            result = int(value) if value else default

            if valid_values and result not in valid_values:
                print(f"  Invalid input. Must be one of: {', '.join(map(str, valid_values))}")
                continue

            return result
        except ValueError:
            print("  Invalid input. Please enter an integer.")


def interactive_calculator():
    """
    Main interactive calculator function

    Guides user through:
    1. Machine parameter input
    2. Current cylinder readings
    3. Target steering state
    4. Generates comprehensive report
    5. Optionally saves report to file
    """

    print("\n" + "="*80)
    print("STEERING CYLINDER CALCULATOR - INTERACTIVE MODE")
    print("="*80 + "\n")

    # ========================================================================
    # STEP 1: Get Machine Parameters
    # ========================================================================

    print("STEP 1: Enter Machine Parameters")
    print("-" * 80)
    print("Press Enter to use default values shown in brackets.\n")

    num_cyl = get_int_input(
        "Number of cylinders",
        default=3,
        valid_values=[3, 4, 6]
    )

    stroke = get_float_input("Cylinder stroke (mm)", default=50.0)
    mount_dia = get_float_input("Mounting diameter (mm)", default=715.0)
    pipe_len = get_float_input("Pipe length (mm)", default=3000.0)
    vert_angle = get_float_input("Vertical angle (mm/m)", default=1.49)

    # Optional parameters (use defaults)
    laser_grad = get_float_input("Laser gradient", default=0.00149)
    dist_head = get_float_input("Distance head to target (mm)", default=2331.0)
    length_head = get_float_input("Length steering head (mm)", default=991.0)
    target_above = get_float_input("Target above axis (mm)", default=140.0)

    params = MachineParameters(
        num_cylinders=num_cyl,
        stroke=stroke,
        mounting_diameter=mount_dia,
        pipe_length=pipe_len,
        vertical_angle=vert_angle,
        laser_gradient=laser_grad,
        dist_head_to_target=dist_head,
        length_steering_head=length_head,
        target_above_axis=target_above
    )

    calc = SteeringCalculator(params)

    # ========================================================================
    # STEP 2: Get Current Cylinder Readings
    # ========================================================================

    print(f"\nSTEP 2: Enter Current Cylinder Readings")
    print("-" * 80)
    print(f"Enter positions for all {num_cyl} cylinders (in mm).\n")

    readings_list = []
    for i in range(1, num_cyl + 1):
        reading = get_float_input(f"Cylinder {i} position (mm)", default=25.0)
        readings_list.append(reading)

    # Create readings object
    cyl_readings = CylinderReadings(
        cylinder_1=readings_list[0],
        cylinder_2=readings_list[1] if len(readings_list) > 1 else 25.0,
        cylinder_3=readings_list[2] if len(readings_list) > 2 else 25.0,
        cylinder_4=readings_list[3] if len(readings_list) > 3 else None,
        cylinder_5=readings_list[4] if len(readings_list) > 4 else None,
        cylinder_6=readings_list[5] if len(readings_list) > 5 else None
    )

    # ========================================================================
    # STEP 3: Analyze Current State
    # ========================================================================

    print("\nAnalyzing current state...")
    analysis = calc.analyze_current_state(cyl_readings)

    current_pitch = analysis['current_steering']['pitch']
    current_yaw = analysis['current_steering']['yaw']

    print(f"\nCurrent Steering State:")
    print(f"  Pitch: {current_pitch:7.2f} mm/m")
    print(f"  Yaw:   {current_yaw:7.2f} mm/m")

    if not analysis['cylinder_status']['all_ok']:
        print("\nWARNINGS:")
        for warning in analysis['cylinder_status']['warnings']:
            print(f"  • {warning}")

    # ========================================================================
    # STEP 4: Get Target Steering State
    # ========================================================================

    print(f"\nSTEP 3: Enter Target Steering State")
    print("-" * 80)
    print("Enter desired pitch and yaw values.\n")

    target_pitch = get_float_input("Target Pitch (mm/m)", default=0.0)
    target_yaw = get_float_input("Target Yaw (mm/m)", default=0.0)

    # ========================================================================
    # STEP 5: Plan Correction
    # ========================================================================

    print("\nCalculating correction plan...")
    correction_plan = calc.plan_correction(
        current_pitch=current_pitch,
        current_yaw=current_yaw,
        target_pitch=target_pitch,
        target_yaw=target_yaw
    )

    # ========================================================================
    # STEP 6: Generate and Display Report
    # ========================================================================

    report = calc.generate_report(analysis, correction_plan)
    print("\n" + report)

    # ========================================================================
    # STEP 7: Save Report (Optional)
    # ========================================================================

    save_choice = input("\nSave report to file? (y/n) [y]: ").strip().lower()
    if save_choice != 'n':
        default_filename = "steering_report.txt"
        filename = input(f"Filename [{default_filename}]: ").strip()
        filename = filename if filename else default_filename

        try:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"\n✓ Report saved to: {filename}")
        except Exception as e:
            print(f"\n✗ Error saving file: {e}")

    # ========================================================================
    # STEP 8: Continue or Exit
    # ========================================================================

    print("\n" + "="*80)
    again = input("Run another calculation? (y/n) [n]: ").strip().lower()
    if again == 'y':
        interactive_calculator()
    else:
        print("\nThank you for using the Steering Calculator!")
        print("="*80 + "\n")


def quick_mode():
    """
    Quick calculation mode with minimal prompts

    For experienced users who want fast calculations with standard parameters.
    """
    print("\n" + "="*80)
    print("STEERING CALCULATOR - QUICK MODE")
    print("="*80 + "\n")

    print("Using standard parameters:")
    print("  • 3 cylinders, 50mm stroke, 715mm diameter, 3000mm pipe length\n")

    mode = input("Calculate (1) Cylinders from Pitch/Yaw or (2) Pitch/Yaw from Cylinders? [1]: ").strip()

    params = MachineParameters()
    calc = SteeringCalculator(params)

    if mode == '2':
        # Reverse calculation
        print("\nEnter cylinder positions:")
        c1 = get_float_input("Cylinder 1 (mm)", 25.0)
        c2 = get_float_input("Cylinder 2 (mm)", 25.0)
        c3 = get_float_input("Cylinder 3 (mm)", 25.0)

        readings = CylinderReadings(cylinder_1=c1, cylinder_2=c2, cylinder_3=c3)
        steering = calc.calculate_steering(readings)

        print("\nRESULT:")
        print(f"  Pitch: {steering.pitch:7.2f} mm/m")
        print(f"  Yaw:   {steering.yaw:7.2f} mm/m")

    else:
        # Forward calculation
        print("\nEnter steering command:")
        pitch = get_float_input("Pitch (mm/m)", 0.0)
        yaw = get_float_input("Yaw (mm/m)", 0.0)

        steering = SteeringCommand(pitch=pitch, yaw=yaw)
        cylinders = calc.calculate_cylinders(steering)

        print("\nRESULT:")
        for cyl, pos in cylinders.items():
            print(f"  {cyl}: {pos:6.2f} mm")

    print("\n" + "="*80 + "\n")


def main():
    """
    Main entry point with mode selection
    """
    print("\n" + "="*80)
    print("MICROTUNNELING STEERING CYLINDER CALCULATOR")
    print("="*80)
    print("\nSelect Mode:")
    print("  1. Interactive Mode (full analysis with reports)")
    print("  2. Quick Mode (fast calculations)")
    print("  3. Exit")

    choice = input("\nEnter choice [1]: ").strip()

    if choice == '2':
        quick_mode()
    elif choice == '3':
        print("\nGoodbye!\n")
        return
    else:
        interactive_calculator()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCalculation cancelled by user.")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("="*80 + "\n")
        raise
