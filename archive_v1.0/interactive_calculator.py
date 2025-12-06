"""
Interactive Steering Cylinder Calculator
==========================================

Complete implementation with report generation and data validation
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class MachineParameters:
    """Machine and tunnel parameters"""
    laser_gradient: float = 0.00149
    vertical_angle: float = 1.49  # mm/m
    dist_head_to_target: float = 2331.0  # mm
    length_steering_head: float = 991.0  # mm
    target_above_axis: float = 140.0  # mm
    num_cylinders: int = 3
    stroke: float = 50.0  # mm
    mounting_diameter: float = 715.0  # mm
    pipe_length: float = 3000.0  # mm


@dataclass
class SteeringCommand:
    """Desired steering corrections"""
    pitch: float = 0.0  # mm/m
    yaw: float = 0.0  # mm/m


@dataclass
class CylinderReadings:
    """Current cylinder position readings"""
    cylinder_1: float = 25.0
    cylinder_2: float = 25.0
    cylinder_3: float = 25.0
    cylinder_4: Optional[float] = None
    cylinder_5: Optional[float] = None
    cylinder_6: Optional[float] = None


class SteeringAnalyzer:
    """
    Complete steering analysis system for microtunneling operations
    """
    
    def __init__(self, params: MachineParameters):
        self.params = params
        self.stroke_center = params.stroke / 2
        self.radius_m = (params.mounting_diameter / 2) / 1000
        
    def analyze_3cylinder(self, readings: CylinderReadings) -> Dict:
        """Complete analysis for 3-cylinder system"""
        
        # Calculate current pitch and yaw from readings
        current_steering = self._calculate_from_3cyl(
            readings.cylinder_1, readings.cylinder_2, readings.cylinder_3
        )
        
        # Calculate drive gradient effects
        gradient_data = self._calculate_gradient()
        
        # Compile analysis
        analysis = {
            'system_type': '3-Cylinder Steering',
            'parameters': asdict(self.params),
            'cylinder_readings': {
                'cylinder_1': readings.cylinder_1,
                'cylinder_2': readings.cylinder_2,
                'cylinder_3': readings.cylinder_3,
            },
            'gradient_analysis': gradient_data,
            'current_steering': {
                'pitch': current_steering.pitch,
                'yaw': current_steering.yaw
            },
            'cylinder_status': self._check_cylinder_status([
                readings.cylinder_1, readings.cylinder_2, readings.cylinder_3
            ])
        }
        
        return analysis
    
    def plan_correction(self, current_pitch: float, current_yaw: float,
                       target_pitch: float, target_yaw: float,
                       num_cylinders: int) -> Dict:
        """
        Plan steering correction to achieve target pitch/yaw
        
        Args:
            current_pitch: Current measured pitch (mm/m)
            current_yaw: Current measured yaw (mm/m)
            target_pitch: Desired target pitch (mm/m)
            target_yaw: Desired target yaw (mm/m)
            num_cylinders: Number of cylinders in system (3, 4, or 6)
            
        Returns:
            Complete correction plan with cylinder positions
        """
        # Calculate required steering correction
        pitch_correction = target_pitch - current_pitch
        yaw_correction = target_yaw - current_yaw
        
        steering_cmd = SteeringCommand(pitch=pitch_correction, yaw=yaw_correction)
        
        # Calculate cylinder positions
        if num_cylinders == 3:
            cylinders = self._calc_3cyl(steering_cmd)
        elif num_cylinders == 4:
            cylinders = self._calc_4cyl(steering_cmd)
        elif num_cylinders == 6:
            cylinders = self._calc_6cyl(steering_cmd)
        else:
            raise ValueError("Number of cylinders must be 3, 4, or 6")
        
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
        
        plan = {
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
            'warnings': self._generate_warnings(cylinders, steering_cmd)
        }
        
        return plan
    
    def generate_report(self, analysis: Dict, correction_plan: Optional[Dict] = None) -> str:
        """Generate a detailed text report"""
        
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
        
        # Correction Plan
        if correction_plan:
            report_lines.append("STEERING CORRECTION PLAN")
            report_lines.append("-" * 80)
            
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
                report_lines.append("⚠️  FEASIBILITY WARNING:")
                report_lines.append(f"  {correction_plan['feasibility']['reason']}")
                report_lines.append("")
            
            if correction_plan['warnings']:
                report_lines.append("⚠️  WARNINGS:")
                for warning in correction_plan['warnings']:
                    report_lines.append(f"  • {warning}")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    # Helper methods
    
    def _calculate_gradient(self) -> Dict:
        """Calculate drive gradient effects"""
        drive_100m = self.params.vertical_angle * 100
        drive_per_pipe = self.params.vertical_angle * (self.params.pipe_length / 1000)
        pitch_from_drive = drive_100m / (self.params.pipe_length / 1000)
        
        return {
            'drive_100m': round(drive_100m, 2),
            'drive_per_pipe': round(drive_per_pipe, 2),
            'pitch_from_drive': round(pitch_from_drive, 2)
        }
    
    def _calc_3cyl(self, steering: SteeringCommand) -> Dict[str, float]:
        """Calculate 3-cylinder positions"""
        pitch_effect = steering.pitch * self.radius_m
        yaw_effect = steering.yaw * self.radius_m
        
        return {
            'cylinder_1': round(self.stroke_center + pitch_effect, 2),
            'cylinder_2': round(self.stroke_center + 
                              (pitch_effect * math.cos(math.radians(120))) +
                              (yaw_effect * math.sin(math.radians(120))), 2),
            'cylinder_3': round(self.stroke_center + 
                              (pitch_effect * math.cos(math.radians(240))) +
                              (yaw_effect * math.sin(math.radians(240))), 2)
        }
    
    def _calc_4cyl(self, steering: SteeringCommand) -> Dict[str, float]:
        """Calculate 4-cylinder positions"""
        pitch_effect = steering.pitch * self.radius_m
        yaw_effect = steering.yaw * self.radius_m
        
        return {
            'cylinder_1': round(self.stroke_center + pitch_effect, 2),
            'cylinder_2': round(self.stroke_center + yaw_effect, 2),
            'cylinder_3': round(self.stroke_center - pitch_effect, 2),
            'cylinder_4': round(self.stroke_center - yaw_effect, 2)
        }
    
    def _calc_6cyl(self, steering: SteeringCommand) -> Dict[str, float]:
        """Calculate 6-cylinder positions"""
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
    
    def _calculate_from_3cyl(self, c1: float, c2: float, c3: float) -> SteeringCommand:
        """Calculate pitch/yaw from 3 cylinder readings"""
        c1_offset = c1 - self.stroke_center
        c2_offset = c2 - self.stroke_center
        c3_offset = c3 - self.stroke_center
        
        pitch = c1_offset / self.radius_m
        yaw = (c2_offset - c3_offset) / (math.sqrt(3) * self.radius_m)
        
        return SteeringCommand(pitch=round(pitch, 2), yaw=round(yaw, 2))
    
    def _check_cylinder_status(self, readings: List[float]) -> Dict:
        """Check cylinder position status"""
        status = []
        for i, reading in enumerate(readings, 1):
            pct = (reading / self.params.stroke) * 100
            if reading < 5:
                status.append(f"Cylinder {i}: NEAR MINIMUM ({reading:.1f}mm, {pct:.1f}%)")
            elif reading > self.params.stroke - 5:
                status.append(f"Cylinder {i}: NEAR MAXIMUM ({reading:.1f}mm, {pct:.1f}%)")
            elif reading < 0 or reading > self.params.stroke:
                status.append(f"Cylinder {i}: OUT OF RANGE ({reading:.1f}mm)")
        
        return {
            'warnings': status,
            'all_ok': len(status) == 0
        }
    
    def _get_position_status(self, position: float) -> str:
        """Get status string for cylinder position"""
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
        """Check if correction is feasible"""
        out_of_range = []
        for cyl, pos in cylinders.items():
            if pos < 0 or pos > self.params.stroke:
                out_of_range.append(f"{cyl}: {pos:.2f}mm")
        
        return {
            'is_feasible': len(out_of_range) == 0,
            'out_of_range': out_of_range,
            'reason': f"Cylinders out of range: {', '.join(out_of_range)}" if out_of_range else "All cylinders within limits"
        }
    
    def _generate_warnings(self, cylinders: Dict[str, float], 
                          steering: SteeringCommand) -> List[str]:
        """Generate warnings for the correction plan"""
        warnings = []
        
        # Check extreme cylinder positions
        for cyl, pos in cylinders.items():
            if pos < 5:
                warnings.append(f"{cyl} very near minimum stroke ({pos:.2f}mm)")
            elif pos > self.params.stroke - 5:
                warnings.append(f"{cyl} very near maximum stroke ({pos:.2f}mm)")
        
        # Check extreme steering angles
        if abs(steering.pitch) > 50:
            warnings.append(f"Very high pitch correction ({steering.pitch:.1f} mm/m)")
        if abs(steering.yaw) > 50:
            warnings.append(f"Very high yaw correction ({steering.yaw:.1f} mm/m)")
        
        return warnings


def interactive_calculator():
    """Interactive command-line calculator"""
    
    print("\n" + "="*80)
    print("STEERING CYLINDER CALCULATOR")
    print("="*80 + "\n")
    
    # Get machine parameters
    print("Enter Machine Parameters (press Enter for defaults):\n")
    
    num_cyl = input("Number of cylinders [3]: ").strip()
    num_cyl = int(num_cyl) if num_cyl else 3
    
    stroke = input("Cylinder stroke (mm) [50]: ").strip()
    stroke = float(stroke) if stroke else 50.0
    
    mount_dia = input("Mounting diameter (mm) [715]: ").strip()
    mount_dia = float(mount_dia) if mount_dia else 715.0
    
    pipe_len = input("Pipe length (mm) [3000]: ").strip()
    pipe_len = float(pipe_len) if pipe_len else 3000.0
    
    vert_angle = input("Vertical angle (mm/m) [1.49]: ").strip()
    vert_angle = float(vert_angle) if vert_angle else 1.49
    
    params = MachineParameters(
        num_cylinders=num_cyl,
        stroke=stroke,
        mounting_diameter=mount_dia,
        pipe_length=pipe_len,
        vertical_angle=vert_angle
    )
    
    analyzer = SteeringAnalyzer(params)
    
    # Get current readings
    print(f"\nEnter Current Cylinder Readings (mm):\n")
    readings = []
    for i in range(1, num_cyl + 1):
        val = input(f"Cylinder {i} [25.0]: ").strip()
        readings.append(float(val) if val else 25.0)
    
    # Create readings object
    cyl_readings = CylinderReadings(
        cylinder_1=readings[0],
        cylinder_2=readings[1] if len(readings) > 1 else 25.0,
        cylinder_3=readings[2] if len(readings) > 2 else 25.0,
        cylinder_4=readings[3] if len(readings) > 3 else None,
        cylinder_5=readings[4] if len(readings) > 4 else None,
        cylinder_6=readings[5] if len(readings) > 5 else None
    )
    
    # Analyze
    if num_cyl == 3:
        analysis = analyzer.analyze_3cylinder(cyl_readings)
    else:
        analysis = {
            'system_type': f'{num_cyl}-Cylinder Steering',
            'parameters': asdict(params),
            'cylinder_readings': {f'cylinder_{i+1}': r for i, r in enumerate(readings)},
            'gradient_analysis': analyzer._calculate_gradient(),
            'current_steering': {'pitch': 0.0, 'yaw': 0.0},
            'cylinder_status': {'warnings': [], 'all_ok': True}
        }
    
    # Get target state
    print("\nEnter Target Steering State:\n")
    target_pitch = input("Target Pitch (mm/m) [0]: ").strip()
    target_pitch = float(target_pitch) if target_pitch else 0.0
    
    target_yaw = input("Target Yaw (mm/m) [0]: ").strip()
    target_yaw = float(target_yaw) if target_yaw else 0.0
    
    # Plan correction
    correction_plan = analyzer.plan_correction(
        current_pitch=analysis['current_steering']['pitch'],
        current_yaw=analysis['current_steering']['yaw'],
        target_pitch=target_pitch,
        target_yaw=target_yaw,
        num_cylinders=num_cyl
    )
    
    # Generate and print report
    report = analyzer.generate_report(analysis, correction_plan)
    print("\n" + report)
    
    # Save report
    save = input("\nSave report to file? (y/n) [y]: ").strip().lower()
    if save != 'n':
        filename = input("Filename [steering_report.txt]: ").strip()
        filename = filename if filename else "steering_report.txt"
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved to {filename}")


if __name__ == "__main__":
    interactive_calculator()
