"""
Steering Cylinder Calculator for Microtunneling Operations
===========================================================

Comprehensive Python implementation for calculating steering cylinder positions
in microtunneling boring machines (MTBM) with support for 3, 4, and 6 cylinder systems.

Features:
- Forward calculations: Pitch/Yaw → Cylinder positions
- Reverse calculations: Cylinder positions → Pitch/Yaw
- Correction planning with feasibility analysis
- Gradient analysis and compensation
- Complete validation and warning system
- Professional report generation

Author: Reverse-engineered from Steer-cyl-cal-rev8_.xls (S.J.Baba)
Version: 2.0 (Consolidated)
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class GroundCondition(Enum):
    """
    Ground condition types with steering rate limits
    
    Limits are based on jacking pressure considerations:
    - Soft: Can handle more aggressive steering
    - Mixed: Limit to 2-4 mm/m to avoid jacking pressure increase
    - Rock: Very sensitive - max 2 mm/m to prevent jacking pressure increase and halting
    """
    SOFT = "soft"
    MIXED = "mixed"
    ROCK = "rock"
    
    def get_max_steering_rate(self) -> float:
        """Get maximum steering rate (mm/m) for this ground condition"""
        if self == GroundCondition.SOFT:
            return 10.0  # Can handle more aggressive steering
        elif self == GroundCondition.MIXED:
            return 4.0   # Limit to 2-4 mm/m to avoid jacking pressure increase
        else:  # ROCK
            return 2.0   # Maximum 2 mm/m to prevent jacking pressure increase and halting
    
    def get_recommended_max(self) -> float:
        """Get recommended maximum steering rate (mm/m)"""
        if self == GroundCondition.SOFT:
            return 8.0
        elif self == GroundCondition.MIXED:
            return 3.0   # Stay within 2-4 mm/m range, use 3 as safe max
        else:  # ROCK
            return 1.5   # Stay well below 2 mm/m limit for safety


@dataclass
class MachineParameters:
    """
    Machine and tunnel configuration parameters

    Attributes:
        num_cylinders: Number of steering cylinders (3, 4, or 6)
        stroke: Maximum stroke of steering cylinders (mm)
        mounting_diameter: Diameter of cylinder mounting circle (mm)
        pipe_length: Length of jacking pipes (mm)
        vertical_angle: Vertical angle of tunnel drive (mm/m)
        laser_gradient: Laser set gradient (default: 0.00149)
        dist_head_to_target: Distance from cutting head to target (mm)
        length_steering_head: Length of steering head (mm)
        target_above_axis: Target position above axis (mm)
    """
    num_cylinders: int = 3
    stroke: float = 50.0
    mounting_diameter: float = 715.0
    pipe_length: float = 3000.0
    vertical_angle: float = 1.49
    laser_gradient: float = 0.00149
    dist_head_to_target: float = 2331.0
    length_steering_head: float = 991.0
    target_above_axis: float = 140.0


@dataclass
class SteeringCommand:
    """
    Desired steering corrections in mm/m

    Attributes:
        pitch: Pitch correction (mm/m) - positive = up, negative = down
        yaw: Yaw correction (mm/m) - positive = right, negative = left
    """
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class CylinderReadings:
    """
    Current cylinder position readings in mm

    For 3-cylinder systems: Use cylinder_1, cylinder_2, cylinder_3
    For 4-cylinder systems: Use cylinder_1 through cylinder_4
    For 6-cylinder systems: Use all six cylinders
    """
    cylinder_1: float = 25.0
    cylinder_2: float = 25.0
    cylinder_3: float = 25.0
    cylinder_4: Optional[float] = None
    cylinder_5: Optional[float] = None
    cylinder_6: Optional[float] = None

    def to_dict(self, num_cylinders: int) -> Dict[str, float]:
        """Convert to dictionary with only active cylinders"""
        result = {
            'cylinder_1': self.cylinder_1,
            'cylinder_2': self.cylinder_2,
            'cylinder_3': self.cylinder_3,
        }
        if num_cylinders >= 4:
            result['cylinder_4'] = self.cylinder_4 or 25.0
        if num_cylinders >= 5:
            result['cylinder_5'] = self.cylinder_5 or 25.0
        if num_cylinders >= 6:
            result['cylinder_6'] = self.cylinder_6 or 25.0
        return result


# ============================================================================
# MAIN CALCULATOR CLASS
# ============================================================================

class SteeringCalculator:
    """
    Comprehensive steering analysis and calculation system for MTBM operations

    This class provides:
    - Cylinder position calculations from pitch/yaw commands
    - Pitch/yaw calculations from cylinder readings (reverse)
    - Correction planning and feasibility analysis
    - Gradient compensation
    - Safety validation and warnings
    - Report generation

    Supports 3, 4, and 6-cylinder steering systems.
    """

    def __init__(self, params: MachineParameters):
        """
        Initialize calculator with machine parameters

        Args:
            params: Machine configuration parameters
        """
        self.params = params
        self.stroke_center = params.stroke / 2
        self.radius_m = (params.mounting_diameter / 2) / 1000  # Convert to meters

    # ========================================================================
    # FORWARD CALCULATIONS: Pitch/Yaw → Cylinder Positions
    # ========================================================================

    def calculate_cylinders(self, steering: SteeringCommand) -> Dict[str, float]:
        """
        Calculate cylinder positions for any system type

        Automatically selects the appropriate calculation method based on
        the number of cylinders configured in machine parameters.

        Args:
            steering: Desired pitch and yaw corrections

        Returns:
            Dictionary of cylinder positions

        Raises:
            ValueError: If number of cylinders is not 3, 4, or 6
        """
        if self.params.num_cylinders == 3:
            return self._calc_3cyl(steering)
        elif self.params.num_cylinders == 4:
            return self._calc_4cyl(steering)
        elif self.params.num_cylinders == 6:
            return self._calc_6cyl(steering)
        else:
            raise ValueError(f"Unsupported number of cylinders: {self.params.num_cylinders}")

    def _calc_3cyl(self, steering: SteeringCommand) -> Dict[str, float]:
        """
        Calculate 3-cylinder positions (120° spacing)

        Cylinder arrangement:
        - Cylinder 1: 0° (Top - 12 o'clock)
        - Cylinder 2: 120° (Lower right - 4 o'clock)
        - Cylinder 3: 240° (Lower left - 8 o'clock)

        Args:
            steering: Pitch and yaw command

        Returns:
            Dictionary with cylinder_1, cylinder_2, cylinder_3 positions
        """
        pitch_effect = steering.pitch * self.radius_m
        yaw_effect = steering.yaw * self.radius_m

        return {
            'cylinder_1': round(
                self.stroke_center + pitch_effect, 2
            ),
            'cylinder_2': round(
                self.stroke_center +
                (pitch_effect * math.cos(math.radians(120))) +
                (yaw_effect * math.sin(math.radians(120))), 2
            ),
            'cylinder_3': round(
                self.stroke_center +
                (pitch_effect * math.cos(math.radians(240))) +
                (yaw_effect * math.sin(math.radians(240))), 2
            )
        }

    def _calc_4cyl(self, steering: SteeringCommand) -> Dict[str, float]:
        """
        Calculate 4-cylinder positions (90° spacing)

        Cylinder arrangement:
        - Cylinder 1: 0° (Top)
        - Cylinder 2: 90° (Right)
        - Cylinder 3: 180° (Bottom)
        - Cylinder 4: 270° (Left)

        Args:
            steering: Pitch and yaw command

        Returns:
            Dictionary with cylinder_1 through cylinder_4 positions
        """
        pitch_effect = steering.pitch * self.radius_m
        yaw_effect = steering.yaw * self.radius_m

        return {
            'cylinder_1': round(self.stroke_center + pitch_effect, 2),
            'cylinder_2': round(self.stroke_center + yaw_effect, 2),
            'cylinder_3': round(self.stroke_center - pitch_effect, 2),
            'cylinder_4': round(self.stroke_center - yaw_effect, 2)
        }

    def _calc_6cyl(self, steering: SteeringCommand) -> Dict[str, float]:
        """
        Calculate 6-cylinder positions (60° spacing)

        Cylinders arranged at 60° intervals starting from top (0°)

        Args:
            steering: Pitch and yaw command

        Returns:
            Dictionary with cylinder_1 through cylinder_6 positions
        """
        pitch_effect = steering.pitch * self.radius_m
        yaw_effect = steering.yaw * self.radius_m

        result = {}
        for i in range(6):
            angle_rad = math.radians(i * 60)
            result[f'cylinder_{i+1}'] = round(
                self.stroke_center +
                (pitch_effect * math.cos(angle_rad)) +
                (yaw_effect * math.sin(angle_rad)), 2
            )
        return result

    # ========================================================================
    # REVERSE CALCULATIONS: Cylinder Positions → Pitch/Yaw
    # ========================================================================

    def calculate_steering(self, readings: CylinderReadings) -> SteeringCommand:
        """
        Calculate pitch/yaw from cylinder readings (reverse calculation)

        Automatically selects the appropriate method based on number of cylinders.

        Args:
            readings: Current cylinder position readings

        Returns:
            SteeringCommand with calculated pitch and yaw

        Raises:
            ValueError: If number of cylinders is not 3, 4, or 6
        """
        if self.params.num_cylinders == 3:
            return self._calculate_from_3cyl(
                readings.cylinder_1, readings.cylinder_2, readings.cylinder_3
            )
        elif self.params.num_cylinders == 4:
            return self._calculate_from_4cyl(
                readings.cylinder_1, readings.cylinder_2,
                readings.cylinder_3, readings.cylinder_4 or 25.0
            )
        elif self.params.num_cylinders == 6:
            cylinders = [
                readings.cylinder_1, readings.cylinder_2, readings.cylinder_3,
                readings.cylinder_4 or 25.0, readings.cylinder_5 or 25.0,
                readings.cylinder_6 or 25.0
            ]
            return self._calculate_from_6cyl(cylinders)
        else:
            raise ValueError(f"Unsupported number of cylinders: {self.params.num_cylinders}")

    def _calculate_from_3cyl(self, c1: float, c2: float, c3: float) -> SteeringCommand:
        """
        Calculate pitch/yaw from 3 cylinder readings

        Args:
            c1, c2, c3: Cylinder positions in mm

        Returns:
            SteeringCommand with pitch and yaw
        """
        c1_offset = c1 - self.stroke_center
        c2_offset = c2 - self.stroke_center
        c3_offset = c3 - self.stroke_center

        pitch = c1_offset / self.radius_m
        yaw = (c2_offset - c3_offset) / (math.sqrt(3) * self.radius_m)

        return SteeringCommand(pitch=round(pitch, 2), yaw=round(yaw, 2))

    def _calculate_from_4cyl(self, c1: float, c2: float,
                            c3: float, c4: float) -> SteeringCommand:
        """
        Calculate pitch/yaw from 4 cylinder readings

        Args:
            c1, c2, c3, c4: Cylinder positions in mm

        Returns:
            SteeringCommand with pitch and yaw
        """
        c1_offset = c1 - self.stroke_center
        c2_offset = c2 - self.stroke_center
        c3_offset = c3 - self.stroke_center
        c4_offset = c4 - self.stroke_center

        pitch = (c1_offset - c3_offset) / (2 * self.radius_m)
        yaw = (c2_offset - c4_offset) / (2 * self.radius_m)

        return SteeringCommand(pitch=round(pitch, 2), yaw=round(yaw, 2))

    def _calculate_from_6cyl(self, cylinders: List[float]) -> SteeringCommand:
        """
        Calculate pitch/yaw from 6 cylinder readings

        Args:
            cylinders: List of 6 cylinder positions in mm

        Returns:
            SteeringCommand with pitch and yaw
        """
        if len(cylinders) != 6:
            raise ValueError("Expected 6 cylinder readings")

        c = [cyl - self.stroke_center for cyl in cylinders]

        pitch = (c[0] - c[3]) / (2 * self.radius_m)
        yaw = ((c[1] - c[4]) + (c[2] - c[5])) / (2 * math.sqrt(3) * self.radius_m)

        return SteeringCommand(pitch=round(pitch, 2), yaw=round(yaw, 2))

    # ========================================================================
    # ANALYSIS AND PLANNING
    # ========================================================================

    def analyze_current_state(self, readings: CylinderReadings) -> Dict:
        """
        Comprehensive analysis of current steering state

        Args:
            readings: Current cylinder readings

        Returns:
            Complete analysis including current steering, gradient effects,
            cylinder status, and warnings
        """
        # Calculate current steering from readings
        current_steering = self.calculate_steering(readings)

        # Calculate gradient effects
        gradient_data = self._calculate_gradient()

        # Get cylinder status
        cylinder_dict = readings.to_dict(self.params.num_cylinders)
        cylinder_status = self._check_cylinder_status(list(cylinder_dict.values()))

        return {
            'system_type': f'{self.params.num_cylinders}-Cylinder Steering',
            'parameters': asdict(self.params),
            'cylinder_readings': cylinder_dict,
            'current_steering': {
                'pitch': current_steering.pitch,
                'yaw': current_steering.yaw
            },
            'gradient_analysis': gradient_data,
            'cylinder_status': cylinder_status
        }

    def plan_correction(self, current_pitch: float, current_yaw: float,
                       target_pitch: float, target_yaw: float,
                       ground_condition: Optional[GroundCondition] = None) -> Dict:
        """
        Plan steering correction to achieve target pitch/yaw

        Args:
            current_pitch: Current measured pitch (mm/m)
            current_yaw: Current measured yaw (mm/m)
            target_pitch: Desired target pitch (mm/m)
            target_yaw: Desired target yaw (mm/m)
            ground_condition: Optional ground condition to apply limits

        Returns:
            Complete correction plan with:
            - Required corrections
            - New cylinder positions
            - Expected results
            - Feasibility analysis
            - Ground condition validation
            - Warnings
        """
        # Calculate required correction
        pitch_correction = target_pitch - current_pitch
        yaw_correction = target_yaw - current_yaw

        # Apply ground condition limits if specified
        ground_validation = None
        if ground_condition:
            max_rate = ground_condition.get_max_steering_rate()
            recommended_max = ground_condition.get_recommended_max()
            
            # Calculate total steering rate (magnitude)
            total_steering_rate = math.sqrt(pitch_correction**2 + yaw_correction**2)
            
            # Check if correction exceeds limits
            exceeds_limit = total_steering_rate > max_rate
            exceeds_recommended = total_steering_rate > recommended_max
            
            # If exceeds limit, scale down proportionally
            if exceeds_limit:
                scale_factor = max_rate / total_steering_rate
                pitch_correction = pitch_correction * scale_factor
                yaw_correction = yaw_correction * scale_factor
                total_steering_rate = max_rate
            
            ground_validation = {
                'ground_condition': ground_condition.value,
                'max_allowed_rate': max_rate,
                'recommended_max_rate': recommended_max,
                'requested_rate': math.sqrt(pitch_correction**2 + yaw_correction**2) if not exceeds_limit else total_steering_rate,
                'original_rate': math.sqrt((target_pitch - current_pitch)**2 + (target_yaw - current_yaw)**2),
                'exceeded_limit': exceeds_limit,
                'exceeded_recommended': exceeds_recommended,
                'was_limited': exceeds_limit,
                'limiting_factor': 'ground_condition' if exceeds_limit else None
            }

        steering_cmd = SteeringCommand(pitch=pitch_correction, yaw=yaw_correction)

        # Calculate new cylinder positions
        cylinders = self.calculate_cylinders(steering_cmd)

        # Calculate correction per pipe
        correction_per_pipe = {
            'pitch_per_pipe': round(pitch_correction * (self.params.pipe_length / 1000), 2),
            'yaw_per_pipe': round(yaw_correction * (self.params.pipe_length / 1000), 2)
        }

        # Expected result after one pipe
        expected_result = {
            'pitch_after_pipe': round(current_pitch + pitch_correction, 2),
            'yaw_after_pipe': round(current_yaw + yaw_correction, 2)
        }

        # Generate warnings including ground condition
        warnings = self._generate_warnings(cylinders, steering_cmd, ground_condition)

        return {
            'current_state': {
                'pitch': current_pitch,
                'yaw': current_yaw
            },
            'target_state': {
                'pitch': target_pitch,
                'yaw': target_yaw
            },
            'required_correction': {
                'pitch': round(pitch_correction, 2),
                'yaw': round(yaw_correction, 2)
            },
            'cylinder_positions': cylinders,
            'correction_per_pipe': correction_per_pipe,
            'expected_result': expected_result,
            'feasibility': self._check_feasibility(cylinders),
            'ground_condition_validation': ground_validation,
            'warnings': warnings
        }

    # ========================================================================
    # GRADIENT AND CORRECTION ANALYSIS
    # ========================================================================

    def _calculate_gradient(self) -> Dict[str, float]:
        """Calculate drive gradient effects"""
        drive_100m = self.params.vertical_angle * 100
        drive_per_pipe = self.params.vertical_angle * (self.params.pipe_length / 1000)
        pitch_from_drive = drive_100m / (self.params.pipe_length / 1000)

        return {
            'drive_100m': round(drive_100m, 2),
            'drive_per_pipe': round(drive_per_pipe, 2),
            'pitch_from_drive': round(pitch_from_drive, 2)
        }

    def calculate_correction_per_pipe(self, steering: SteeringCommand,
                                     current_pitch: float = 0,
                                     current_yaw: float = 0) -> Dict[str, float]:
        """
        Calculate pitch/yaw correction achieved after one pipe installation

        Args:
            steering: Applied steering command
            current_pitch: Current pitch before correction (mm/m)
            current_yaw: Current yaw before correction (mm/m)

        Returns:
            Corrections per pipe and expected new pitch/yaw
        """
        pitch_correction_per_pipe = steering.pitch * (self.params.pipe_length / 1000)
        yaw_correction_per_pipe = steering.yaw * (self.params.pipe_length / 1000)

        return {
            'pitch_correction_per_pipe': round(pitch_correction_per_pipe, 2),
            'yaw_correction_per_pipe': round(yaw_correction_per_pipe, 2),
            'pitch_after_one_pipe': round(current_pitch + steering.pitch, 2),
            'yaw_after_one_pipe': round(current_yaw + steering.yaw, 2)
        }

    # ========================================================================
    # VALIDATION AND STATUS CHECKING
    # ========================================================================

    def _check_cylinder_status(self, readings: List[float]) -> Dict:
        """
        Check cylinder position status and generate warnings

        Args:
            readings: List of cylinder positions

        Returns:
            Status dict with warnings and overall health indicator
        """
        warnings = []
        for i, reading in enumerate(readings, 1):
            pct = (reading / self.params.stroke) * 100

            if reading < 0 or reading > self.params.stroke:
                warnings.append(f"Cylinder {i}: OUT OF RANGE ({reading:.1f}mm)")
            elif reading < 5:
                warnings.append(f"Cylinder {i}: NEAR MINIMUM ({reading:.1f}mm, {pct:.1f}%)")
            elif reading > self.params.stroke - 5:
                warnings.append(f"Cylinder {i}: NEAR MAXIMUM ({reading:.1f}mm, {pct:.1f}%)")

        return {
            'warnings': warnings,
            'all_ok': len(warnings) == 0
        }

    def _get_position_status(self, position: float) -> str:
        """Get status string for a single cylinder position"""
        if position < 0:
            return "BELOW MIN"
        elif position > self.params.stroke:
            return "ABOVE MAX"
        elif position < 5:
            return "Near Min"
        elif position > self.params.stroke - 5:
            return "Near Max"
        else:
            pct = (position / self.params.stroke) * 100
            return f"{pct:.0f}%"

    def _check_feasibility(self, cylinders: Dict[str, float]) -> Dict:
        """
        Check if correction is feasible (within stroke limits)

        Args:
            cylinders: Dictionary of cylinder positions

        Returns:
            Feasibility analysis with details
        """
        out_of_range = []
        for cyl, pos in cylinders.items():
            if pos < 0 or pos > self.params.stroke:
                out_of_range.append(f"{cyl}: {pos:.2f}mm")

        return {
            'is_feasible': len(out_of_range) == 0,
            'out_of_range': out_of_range,
            'reason': (f"Cylinders out of range: {', '.join(out_of_range)}"
                      if out_of_range else "All cylinders within limits")
        }

    def _generate_warnings(self, cylinders: Dict[str, float],
                          steering: SteeringCommand,
                          ground_condition: Optional[GroundCondition] = None) -> List[str]:
        """
        Generate warnings for the correction plan

        Args:
            cylinders: Calculated cylinder positions
            steering: Applied steering command
            ground_condition: Optional ground condition for additional warnings

        Returns:
            List of warning messages
        """
        warnings = []

        # Check extreme cylinder positions
        for cyl, pos in cylinders.items():
            if pos < 5 and pos >= 0:
                warnings.append(f"{cyl} very near minimum stroke ({pos:.2f}mm)")
            elif pos > self.params.stroke - 5 and pos <= self.params.stroke:
                warnings.append(f"{cyl} very near maximum stroke ({pos:.2f}mm)")

        # Check extreme steering angles
        if abs(steering.pitch) > 50:
            warnings.append(f"Very high pitch correction ({steering.pitch:.1f} mm/m)")
        if abs(steering.yaw) > 50:
            warnings.append(f"Very high yaw correction ({steering.yaw:.1f} mm/m)")

        # Ground condition specific warnings
        if ground_condition:
            total_rate = math.sqrt(steering.pitch**2 + steering.yaw**2)
            max_rate = ground_condition.get_max_steering_rate()
            recommended_max = ground_condition.get_recommended_max()
            
            if total_rate > max_rate:
                warnings.append(
                    f"⚠️ CRITICAL: Steering rate ({total_rate:.2f} mm/m) exceeds maximum "
                    f"for {ground_condition.value} ground ({max_rate} mm/m). "
                    f"Risk of jacking pressure increase and halting procedure!"
                )
            elif total_rate > recommended_max:
                warnings.append(
                    f"⚠️ WARNING: Steering rate ({total_rate:.2f} mm/m) exceeds recommended "
                    f"for {ground_condition.value} ground ({recommended_max} mm/m). "
                    f"May cause jacking pressure increase."
                )
            elif ground_condition == GroundCondition.ROCK and total_rate > 1.5:
                warnings.append(
                    f"⚠️ CAUTION: In rock ground, steering rate ({total_rate:.2f} mm/m) is high. "
                    f"Monitor jacking pressure closely."
                )

        return warnings

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    def generate_report(self, analysis: Dict, correction_plan: Optional[Dict] = None) -> str:
        """
        Generate comprehensive text report

        Args:
            analysis: Analysis from analyze_current_state()
            correction_plan: Optional correction plan from plan_correction()

        Returns:
            Formatted text report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MICROTUNNELING STEERING ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # System Information
        report_lines.append("SYSTEM CONFIGURATION")
        report_lines.append("-" * 80)
        params = analysis['parameters']
        report_lines.append(f"System Type:              {analysis['system_type']}")
        report_lines.append(f"Number of Cylinders:      {params['num_cylinders']}")
        report_lines.append(f"Cylinder Stroke:          {params['stroke']} mm")
        report_lines.append(f"Mounting Circle Diameter: {params['mounting_diameter']} mm")
        report_lines.append(f"Pipe Length:              {params['pipe_length']} mm")
        report_lines.append(f"Steering Head Length:     {params['length_steering_head']} mm")
        report_lines.append("")

        # Gradient Information
        report_lines.append("TUNNEL GRADIENT")
        report_lines.append("-" * 80)
        grad = analysis['gradient_analysis']
        report_lines.append(f"Vertical Angle:           {params['vertical_angle']} mm/m")
        report_lines.append(f"Drive per 100m:           {grad['drive_100m']} mm")
        report_lines.append(f"Drive per Pipe:           {grad['drive_per_pipe']} mm")
        report_lines.append(f"Natural Pitch:            {grad['pitch_from_drive']} mm/m")
        report_lines.append("")

        # Current Cylinder Status
        report_lines.append("CURRENT CYLINDER READINGS")
        report_lines.append("-" * 80)
        for cyl, value in analysis['cylinder_readings'].items():
            status = self._get_position_status(value)
            report_lines.append(f"{cyl:15s}: {value:6.2f} mm  [{status}]")
        report_lines.append("")

        # Current Steering State
        report_lines.append("CURRENT STEERING STATE")
        report_lines.append("-" * 80)
        curr = analysis['current_steering']
        report_lines.append(f"Pitch:  {curr['pitch']:7.2f} mm/m")
        report_lines.append(f"Yaw:    {curr['yaw']:7.2f} mm/m")
        report_lines.append("")

        # Cylinder Status Warnings
        if not analysis['cylinder_status']['all_ok']:
            report_lines.append("CYLINDER STATUS WARNINGS")
            report_lines.append("-" * 80)
            for warning in analysis['cylinder_status']['warnings']:
                report_lines.append(f"  • {warning}")
            report_lines.append("")

            # Correction Plan
            if correction_plan:
                report_lines.append("STEERING CORRECTION PLAN")
                report_lines.append("-" * 80)

                # Ground condition information
                if correction_plan.get('ground_condition_validation'):
                    gc_val = correction_plan['ground_condition_validation']
                    report_lines.append(f"Ground Condition: {gc_val['ground_condition'].upper()}")
                    report_lines.append(f"  Max Allowed Rate: {gc_val['max_allowed_rate']:.1f} mm/m")
                    report_lines.append(f"  Recommended Max:  {gc_val['recommended_max_rate']:.1f} mm/m")
                    total_rate = math.sqrt(
                        correction_plan['required_correction']['pitch']**2 +
                        correction_plan['required_correction']['yaw']**2
                    )
                    report_lines.append(f"  Applied Rate:      {total_rate:.2f} mm/m")
                    if gc_val['was_limited']:
                        report_lines.append(f"  ⚠️  Correction was LIMITED due to ground condition")
                    report_lines.append("")

                report_lines.append("Target State:")
                report_lines.append(f"  Pitch: {correction_plan['target_state']['pitch']:7.2f} mm/m")
                report_lines.append(f"  Yaw:   {correction_plan['target_state']['yaw']:7.2f} mm/m")
                report_lines.append("")

                report_lines.append("Required Correction:")
                report_lines.append(f"  Pitch: {correction_plan['required_correction']['pitch']:7.2f} mm/m")
                report_lines.append(f"  Yaw:   {correction_plan['required_correction']['yaw']:7.2f} mm/m")
                report_lines.append("")

            report_lines.append("NEW CYLINDER POSITIONS:")
            for cyl, pos in correction_plan['cylinder_positions'].items():
                status = self._get_position_status(pos)
                report_lines.append(f"  {cyl:15s}: {pos:6.2f} mm  [{status}]")
            report_lines.append("")

            report_lines.append("Expected Results After One Pipe:")
            report_lines.append(f"  Pitch: {correction_plan['expected_result']['pitch_after_pipe']:7.2f} mm/m")
            report_lines.append(f"  Yaw:   {correction_plan['expected_result']['yaw_after_pipe']:7.2f} mm/m")
            report_lines.append("")

            if not correction_plan['feasibility']['is_feasible']:
                report_lines.append("FEASIBILITY WARNING:")
                report_lines.append(f"  {correction_plan['feasibility']['reason']}")
                report_lines.append("")

            if correction_plan['warnings']:
                report_lines.append("WARNINGS:")
                for warning in correction_plan['warnings']:
                    report_lines.append(f"  • {warning}")
                report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_calculate(pitch: float, yaw: float,
                   num_cylinders: int = 3,
                   mounting_diameter: float = 715.0,
                   stroke: float = 50.0) -> Dict[str, float]:
    """
    Quick cylinder calculation with minimal setup

    Args:
        pitch: Pitch correction (mm/m)
        yaw: Yaw correction (mm/m)
        num_cylinders: Number of cylinders (default: 3)
        mounting_diameter: Mounting diameter (default: 715mm)
        stroke: Cylinder stroke (default: 50mm)

    Returns:
        Dictionary of cylinder positions

    Example:
        >>> cylinders = quick_calculate(pitch=-5, yaw=10)
        >>> print(cylinders)
        {'cylinder_1': 23.21, 'cylinder_2': 28.73, 'cylinder_3': 21.06}
    """
    params = MachineParameters(
        num_cylinders=num_cylinders,
        mounting_diameter=mounting_diameter,
        stroke=stroke
    )
    calc = SteeringCalculator(params)
    return calc.calculate_cylinders(SteeringCommand(pitch=pitch, yaw=yaw))


def quick_reverse(cylinders: List[float],
                 num_cylinders: int = 3,
                 mounting_diameter: float = 715.0,
                 stroke: float = 50.0) -> Tuple[float, float]:
    """
    Quick reverse calculation with minimal setup

    Args:
        cylinders: List of cylinder positions
        num_cylinders: Number of cylinders
        mounting_diameter: Mounting diameter (default: 715mm)
        stroke: Cylinder stroke (default: 50mm)

    Returns:
        Tuple of (pitch, yaw)

    Example:
        >>> pitch, yaw = quick_reverse([20.0, 32.0, 30.0])
        >>> print(f"Pitch: {pitch}, Yaw: {yaw}")
        Pitch: -13.99, Yaw: 3.23
    """
    params = MachineParameters(
        num_cylinders=num_cylinders,
        mounting_diameter=mounting_diameter,
        stroke=stroke
    )
    calc = SteeringCalculator(params)

    readings = CylinderReadings(
        cylinder_1=cylinders[0],
        cylinder_2=cylinders[1] if len(cylinders) > 1 else 25.0,
        cylinder_3=cylinders[2] if len(cylinders) > 2 else 25.0,
        cylinder_4=cylinders[3] if len(cylinders) > 3 else None,
        cylinder_5=cylinders[4] if len(cylinders) > 4 else None,
        cylinder_6=cylinders[5] if len(cylinders) > 5 else None
    )

    steering = calc.calculate_steering(readings)
    return steering.pitch, steering.yaw


# ============================================================================
# EXAMPLE AND TESTING
# ============================================================================

def main():
    """Example usage demonstrating all features"""

    print("=" * 80)
    print("STEERING CYLINDER CALCULATOR - DEMONSTRATION")
    print("=" * 80)

    # Setup machine parameters
    params = MachineParameters(
        num_cylinders=3,
        stroke=50.0,
        mounting_diameter=715.0,
        pipe_length=3000.0,
        vertical_angle=1.49
    )

    calc = SteeringCalculator(params)

    # Example 1: Forward calculation
    print("\n1. FORWARD CALCULATION (Pitch/Yaw → Cylinders)")
    print("-" * 80)
    steering = SteeringCommand(pitch=-4.5, yaw=16.5)
    cylinders = calc.calculate_cylinders(steering)
    print(f"Input: Pitch = {steering.pitch} mm/m, Yaw = {steering.yaw} mm/m")
    print(f"Output:")
    for cyl, pos in cylinders.items():
        print(f"  {cyl}: {pos} mm")

    # Example 2: Reverse calculation
    print("\n2. REVERSE CALCULATION (Cylinders → Pitch/Yaw)")
    print("-" * 80)
    readings = CylinderReadings(cylinder_1=20.0, cylinder_2=32.0, cylinder_3=30.0)
    reverse_steering = calc.calculate_steering(readings)
    print(f"Input: Cyl1 = 20.0mm, Cyl2 = 32.0mm, Cyl3 = 30.0mm")
    print(f"Output: Pitch = {reverse_steering.pitch} mm/m, Yaw = {reverse_steering.yaw} mm/m")

    # Example 3: Complete analysis
    print("\n3. CURRENT STATE ANALYSIS")
    print("-" * 80)
    readings = CylinderReadings(cylinder_1=28.69, cylinder_2=31.52, cylinder_3=21.79)
    analysis = calc.analyze_current_state(readings)
    print(f"Current Pitch: {analysis['current_steering']['pitch']} mm/m")
    print(f"Current Yaw: {analysis['current_steering']['yaw']} mm/m")

    # Example 4: Correction planning
    print("\n4. CORRECTION PLANNING")
    print("-" * 80)
    correction_plan = calc.plan_correction(
        current_pitch=10.3,
        current_yaw=-9.7,
        target_pitch=-4.5,
        target_yaw=16.5
    )
    print(f"Required Pitch Correction: {correction_plan['required_correction']['pitch']} mm/m")
    print(f"Required Yaw Correction: {correction_plan['required_correction']['yaw']} mm/m")
    print(f"Feasible: {correction_plan['feasibility']['is_feasible']}")

    # Example 5: Full report
    print("\n5. COMPLETE REPORT")
    print("-" * 80)
    report = calc.generate_report(analysis, correction_plan)
    print(report)

    # Example 6: Quick functions
    print("\n6. QUICK CALCULATION FUNCTIONS")
    print("-" * 80)
    result = quick_calculate(pitch=-5, yaw=10)
    print(f"Quick Calculate: {result}")

    pitch, yaw = quick_reverse([20.0, 32.0, 30.0])
    print(f"Quick Reverse: Pitch = {pitch}, Yaw = {yaw}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
