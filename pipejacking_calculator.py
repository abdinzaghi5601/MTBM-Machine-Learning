"""
Hitchhiker's Guide to Pipejacking - Calculator Implementation
=============================================================
Python implementation of the Excel calculator by Lutz Henke

Replicates all calculations from the Excel file:
- Push Power & Jacking Speed
- Slurry Flow Rates
- Power Requirements
- Cable Sizing
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class JackingStationParams:
    """Parameters for jacking station"""
    num_cylinders: int
    cylinder_diameter_mm: float
    oil_flow_lpm: float  # L/min
    max_pressure_bar: float


@dataclass
class SlurryParams:
    """Parameters for slurry system"""
    pipe_diameter_mm: float


@dataclass
class PowerParams:
    """Parameters for power calculations"""
    power_factor: float = 0.8
    voltage: float = 400.0  # Volts


@dataclass
class JackingResults:
    """Results from jacking calculations"""
    combined_area_cm2: float
    total_force_tonnes: float
    max_speed_cm_per_min: float
    time_per_pipe_minutes: float


@dataclass
class PipeParams:
    """Pipe parameters for various calculations"""
    outer_diameter_mm: float      # Pipe outer diameter (Da)
    inner_diameter_mm: float      # Pipe inner diameter (di)
    length_m: float = 3.0         # Pipe length (default 3m)
    concrete_density: float = 2.42  # Reinforced concrete density (t/m³)


@dataclass
class TunnelParams:
    """Tunnel/drive parameters"""
    cutter_diameter_mm: float     # Cutterhead diameter with overcut (Dü)
    pipe_outer_diameter_mm: float # Pipe outer diameter (Da)
    total_distance_m: float       # Total tunnel distance
    machine_length_m: float = 10.0  # TBM length


@dataclass
class ExpanderConfig:
    """Expander/Interjack configuration"""
    position_m: float             # Distance from machine
    force_tonnes: float           # Force applied
    name: str = ""                # Identifier


class PipejackingCalculator:
    """
    Main calculator class for pipejacking operations
    """
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    def calculate_jacking_force(self, 
                               params: JackingStationParams) -> JackingResults:
        """
        Calculate jacking force and speed for a jacking station.
        
        Args:
            params: JackingStationParams with station parameters
        
        Returns:
            JackingResults with calculated values
        """
        # Calculate combined full bore area
        radius_cm = (params.cylinder_diameter_mm / 2) / 10  # Convert mm to cm
        area_per_cylinder_cm2 = math.pi * radius_cm ** 2
        combined_area_cm2 = area_per_cylinder_cm2 * params.num_cylinders
        
        # Calculate total force
        # Force = Pressure × Area
        # 1 bar = 1 kg/cm², convert to tonnes (divide by 1000)
        total_force_tonnes = (params.max_pressure_bar * combined_area_cm2) / 1000
        
        # Calculate maximum speed
        # Speed = Oil Flow / Area
        # Convert L/min to cm³/min: 1 L = 1000 cm³
        oil_flow_cm3_per_min = params.oil_flow_lpm * 1000
        max_speed_cm_per_min = oil_flow_cm3_per_min / combined_area_cm2
        
        # Calculate time to push one pipe (assuming 3m pipe)
        pipe_length_cm = 300  # 3 meters = 300 cm
        time_per_pipe_minutes = pipe_length_cm / max_speed_cm_per_min
        
        return JackingResults(
            combined_area_cm2=combined_area_cm2,
            total_force_tonnes=total_force_tonnes,
            max_speed_cm_per_min=max_speed_cm_per_min,
            time_per_pipe_minutes=time_per_pipe_minutes
        )
    
    def calculate_slurry_flow_rate(self, 
                                  params: SlurryParams,
                                  velocity_mps: float) -> Dict[str, float]:
        """
        Calculate slurry flow rate for given velocity.
        
        Args:
            params: SlurryParams with pipe diameter
            velocity_mps: Velocity in meters per second
        
        Returns:
            Dictionary with flow rates in different units
        """
        # Calculate cross-sectional area
        radius_m = (params.pipe_diameter_mm / 2) / 1000  # Convert mm to m
        area_m2 = math.pi * radius_m ** 2
        
        # Flow rate in m³/s
        flow_rate_m3_per_s = area_m2 * velocity_mps
        
        # Convert to different units
        flow_rate_m3_per_hr = flow_rate_m3_per_s * 3600
        flow_rate_l_per_min = flow_rate_m3_per_s * 60 * 1000
        
        return {
            'velocity_mps': velocity_mps,
            'area_m2': area_m2,
            'flow_rate_m3_per_s': flow_rate_m3_per_s,
            'flow_rate_m3_per_hr': flow_rate_m3_per_hr,
            'flow_rate_l_per_min': flow_rate_l_per_min
        }
    
    def generate_slurry_flow_table(self, 
                                   params: SlurryParams,
                                   velocities: List[float] = None) -> List[Dict]:
        """
        Generate flow rate table for different velocities.
        
        Args:
            params: SlurryParams with pipe diameter
            velocities: List of velocities (default: 3.0 to 5.0 m/s in 0.25 steps)
        
        Returns:
            List of flow rate dictionaries
        """
        if velocities is None:
            velocities = [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
        
        results = []
        for vel in velocities:
            flow_data = self.calculate_slurry_flow_rate(params, vel)
            results.append(flow_data)
        
        return results
    
    def calculate_power_requirements(self,
                                    motor_powers_kw: List[float],
                                    power_params: PowerParams) -> Dict[str, float]:
        """
        Calculate total power requirements.
        
        Args:
            motor_powers_kw: List of motor powers in kW
            power_params: PowerParams with power factor and voltage
        
        Returns:
            Dictionary with power calculations
        """
        total_kw = sum(motor_powers_kw)
        total_kva = total_kw / power_params.power_factor
        generator_size_kva = total_kva * 0.8  # 80% of total
        
        return {
            'total_kw': total_kw,
            'total_kva': total_kva,
            'generator_size_kva': generator_size_kva,
            'power_factor': power_params.power_factor
        }
    
    def calculate_cable_amperage(self,
                                 kva: float,
                                 voltage: float,
                                 three_phase: bool = True) -> float:
        """
        Calculate amperage from kVA and voltage.
        
        Args:
            kva: Power in kVA
            voltage: Voltage in Volts
            three_phase: Whether 3-phase (default: True)
        
        Returns:
            Amperage in Amps
        """
        if three_phase:
            # 3-phase: Amps = (kVA × 1000) / (Voltage × √3)
            amps = (kva * 1000) / (voltage * math.sqrt(3))
        else:
            # Single-phase: Amps = (kVA × 1000) / Voltage
            amps = (kva * 1000) / voltage
        
        return amps
    
    def get_cable_size(self, amps: float) -> Dict[str, any]:
        """
        Get recommended cable size from amperage.
        
        Uses Table 4E2A (BS5467 & BS7211) values.
        
        Args:
            amps: Required amperage
        
        Returns:
            Dictionary with cable size recommendation
        """
        # Cable sizing table (from Excel)
        cable_table = [
            (1, 18), (1.5, 23), (2.5, 32), (4, 42), (6, 54),
            (10, 75), (16, 100), (25, 127), (35, 158), (50, 192),
            (70, 246), (95, 298), (120, 346), (150, 399),
            (185, 456), (240, 538), (300, 621), (400, 741)
        ]
        
        # Find appropriate cable size
        recommended_size = None
        for size_mm2, rating_amps in cable_table:
            if amps <= rating_amps:
                recommended_size = size_mm2
                break
        
        if recommended_size is None:
            recommended_size = 400  # Maximum in table
            rating_amps = 741
        
        return {
            'required_amps': amps,
            'recommended_size_mm2': recommended_size,
            'cable_rating_amps': rating_amps,
            'safety_margin': rating_amps - amps
        }

    # =========================================================================
    # NEW CALCULATIONS FROM EXERCISE EXAMPLES
    # =========================================================================

    def calculate_overcut_volume(self,
                                 cutter_diameter_mm: float,
                                 pipe_outer_diameter_mm: float,
                                 length_m: float = 1.0) -> Dict:
        """
        Calculate theoretical overcut (annular) volume.

        Formula: V_ü = π/4 × (D_cutter² - D_pipe²) × L

        Args:
            cutter_diameter_mm: Cutterhead diameter with overcut (mm)
            pipe_outer_diameter_mm: Pipe outer diameter (mm)
            length_m: Length to calculate for (default 1m = per running meter)

        Returns:
            Dictionary with overcut calculations
        """
        D_cutter = cutter_diameter_mm / 1000  # Convert to m
        D_pipe = pipe_outer_diameter_mm / 1000

        # Overcut per side
        overcut_mm = (cutter_diameter_mm - pipe_outer_diameter_mm) / 2

        # Areas
        area_cutter = math.pi / 4 * D_cutter ** 2
        area_pipe = math.pi / 4 * D_pipe ** 2
        area_overcut = area_cutter - area_pipe

        # Volume
        volume_m3 = area_overcut * length_m
        volume_L = volume_m3 * 1000

        return {
            'cutter_diameter_mm': cutter_diameter_mm,
            'pipe_outer_diameter_mm': pipe_outer_diameter_mm,
            'overcut_per_side_mm': overcut_mm,
            'overcut_area_m2': area_overcut,
            'volume_per_m_m3': volume_m3 / length_m,
            'volume_per_m_L': volume_L / length_m,
            'total_volume_m3': volume_m3,
            'total_volume_L': volume_L,
            'length_m': length_m,
        }

    def calculate_bentonite_flow_for_overcut(self,
                                             overcut_volume_m3_per_m: float,
                                             advance_speed_cm_min: float,
                                             loss_factor: float = 1.15) -> Dict:
        """
        Calculate bentonite suspension flow rate to fill overcut.

        Args:
            overcut_volume_m3_per_m: Overcut volume per running meter (m³/lfm)
            advance_speed_cm_min: Advance speed (cm/min)
            loss_factor: Factor for ground loss (default 1.15 = 15% loss)

        Returns:
            Dictionary with bentonite flow requirements
        """
        # Time to advance 1 meter
        time_per_m_min = 100 / advance_speed_cm_min  # 100 cm = 1 m

        # Base flow rate to fill overcut
        flow_m3_min = overcut_volume_m3_per_m / time_per_m_min
        flow_L_min = flow_m3_min * 1000

        # With loss factor
        flow_with_loss_m3_min = flow_m3_min * loss_factor
        flow_with_loss_L_min = flow_L_min * loss_factor

        return {
            'overcut_volume_m3_per_m': overcut_volume_m3_per_m,
            'advance_speed_cm_min': advance_speed_cm_min,
            'time_per_meter_min': time_per_m_min,
            'base_flow_m3_min': flow_m3_min,
            'base_flow_L_min': flow_L_min,
            'loss_factor': loss_factor,
            'loss_percent': (loss_factor - 1) * 100,
            'total_flow_m3_min': flow_with_loss_m3_min,
            'total_flow_L_min': flow_with_loss_L_min,
        }

    def calculate_lubrication_stations(self,
                                       total_distance_m: float,
                                       machine_length_m: float,
                                       pipes_between_stations: int,
                                       pipe_length_m: float,
                                       volume_per_station_L: float,
                                       advance_speed_cm_min: float) -> Dict:
        """
        Calculate lubrication station requirements.

        Args:
            total_distance_m: Total tunnel distance (m)
            machine_length_m: TBM length (m)
            pipes_between_stations: Number of standard pipes between lubrication pipes
            pipe_length_m: Length of each pipe (m)
            volume_per_station_L: Bentonite volume per station per pipe advance (L)
            advance_speed_cm_min: Advance speed (cm/min)

        Returns:
            Dictionary with lubrication station calculations
        """
        # Pipe section length (excluding machine)
        pipe_section_length = total_distance_m - machine_length_m

        # Distance between lubrication stations
        station_spacing_m = pipes_between_stations * pipe_length_m

        # Number of stations
        num_stations = int(pipe_section_length / station_spacing_m)

        # Total volume per pipe advance
        total_volume_per_pipe_L = num_stations * volume_per_station_L
        total_volume_per_pipe_m3 = total_volume_per_pipe_L / 1000

        # Time to advance one pipe
        time_per_pipe_min = (pipe_length_m * 100) / advance_speed_cm_min

        # Flow rate for lubrication
        lub_flow_L_min = total_volume_per_pipe_L / time_per_pipe_min

        return {
            'total_distance_m': total_distance_m,
            'machine_length_m': machine_length_m,
            'pipe_section_length_m': pipe_section_length,
            'station_spacing_m': station_spacing_m,
            'num_stations': num_stations,
            'volume_per_station_L': volume_per_station_L,
            'total_volume_per_pipe_L': total_volume_per_pipe_L,
            'total_volume_per_pipe_m3': total_volume_per_pipe_m3,
            'time_per_pipe_min': time_per_pipe_min,
            'lubrication_flow_L_min': lub_flow_L_min,
        }

    def calculate_excavation_volume(self,
                                    cutter_diameter_mm: float,
                                    pipe_length_m: float,
                                    swell_factor: float = 1.2,
                                    separation_efficiency: float = 0.95) -> Dict:
        """
        Calculate excavation volume per pipe with swell factor.

        Args:
            cutter_diameter_mm: Cutterhead diameter (mm)
            pipe_length_m: Pipe length (m)
            swell_factor: Volume increase when excavated (default 1.2)
            separation_efficiency: Fraction separated (default 0.95 = 95%)

        Returns:
            Dictionary with excavation volumes
        """
        D = cutter_diameter_mm / 1000  # Convert to m

        # In-situ volume
        insitu_volume = math.pi / 4 * D ** 2 * pipe_length_m

        # Bulked/swelled volume
        bulked_volume = insitu_volume * swell_factor

        # Volume delivered to separation (accounting for losses)
        separated_volume = bulked_volume * separation_efficiency

        return {
            'cutter_diameter_mm': cutter_diameter_mm,
            'pipe_length_m': pipe_length_m,
            'insitu_volume_m3': insitu_volume,
            'swell_factor': swell_factor,
            'bulked_volume_m3': bulked_volume,
            'separation_efficiency': separation_efficiency,
            'separated_volume_m3': separated_volume,
            'loss_volume_m3': bulked_volume - separated_volume,
        }

    def calculate_ground_mass(self,
                              volume_m3: float,
                              material_density_t_m3: float = 1.9) -> Dict:
        """
        Calculate ground mass from volume.

        Args:
            volume_m3: Volume of material (m³)
            material_density_t_m3: Material density (t/m³), default 1.9

        Returns:
            Dictionary with mass calculations
        """
        mass_tonnes = volume_m3 * material_density_t_m3
        mass_kg = mass_tonnes * 1000

        return {
            'volume_m3': volume_m3,
            'material_density_t_m3': material_density_t_m3,
            'mass_tonnes': mass_tonnes,
            'mass_kg': mass_kg,
        }

    def calculate_buoyancy(self, pipe: PipeParams) -> Dict:
        """
        Calculate pipe buoyancy under groundwater.

        Buoyancy = Mass of pipe - Mass of displaced water
        Negative result = buoyancy (pipe floats)
        Positive result = no buoyancy (pipe sinks)

        Args:
            pipe: PipeParams with pipe dimensions

        Returns:
            Dictionary with buoyancy calculations
        """
        D_outer = pipe.outer_diameter_mm / 1000  # m
        D_inner = pipe.inner_diameter_mm / 1000  # m
        L = pipe.length_m
        rho_concrete = pipe.concrete_density  # t/m³
        rho_water = 1.0  # t/m³

        # Volume of displaced water (outer volume)
        V_displaced = math.pi / 4 * D_outer ** 2 * L
        mass_water = V_displaced * rho_water

        # Volume of concrete (pipe wall)
        V_outer = math.pi / 4 * D_outer ** 2 * L
        V_inner = math.pi / 4 * D_inner ** 2 * L
        V_concrete = V_outer - V_inner
        mass_pipe = V_concrete * rho_concrete

        # Buoyancy force (negative = floats)
        buoyancy_force = mass_pipe - mass_water

        # Status
        if buoyancy_force < 0:
            status = "BUOYANT - Pipe will float (positioned on roof)"
            is_buoyant = True
        else:
            status = "NOT BUOYANT - Pipe rests in invert"
            is_buoyant = False

        return {
            'pipe_outer_diameter_mm': pipe.outer_diameter_mm,
            'pipe_inner_diameter_mm': pipe.inner_diameter_mm,
            'pipe_length_m': pipe.length_m,
            'concrete_density_t_m3': pipe.concrete_density,
            'volume_displaced_m3': V_displaced,
            'mass_water_displaced_t': mass_water,
            'volume_concrete_m3': V_concrete,
            'mass_pipe_t': mass_pipe,
            'buoyancy_force_t': buoyancy_force,
            'is_buoyant': is_buoyant,
            'status': status,
        }

    def calculate_groundwater_force(self,
                                    cutter_diameter_mm: float,
                                    water_pressure_bar: float) -> Dict:
        """
        Calculate groundwater force on tunnel face.

        Formula: F = A × P × 10 (converts bar to tonnes)

        Args:
            cutter_diameter_mm: Cutterhead diameter (mm)
            water_pressure_bar: Groundwater pressure at face (bar)

        Returns:
            Dictionary with groundwater force
        """
        D = cutter_diameter_mm / 1000  # m
        A = math.pi / 4 * D ** 2  # m²

        # Force in tonnes (1 bar × 1 m² × 10 = 10 tonnes)
        force_tonnes = A * water_pressure_bar * 10
        force_kN = force_tonnes * 9.81

        return {
            'cutter_diameter_mm': cutter_diameter_mm,
            'face_area_m2': A,
            'water_pressure_bar': water_pressure_bar,
            'force_tonnes': force_tonnes,
            'force_kN': force_kN,
            'note': 'Machine should be secured when starting advance!',
        }

    def calculate_specific_friction(self,
                                    force_tonnes: float,
                                    pipe_diameter_mm: float,
                                    section_length_m: float) -> Dict:
        """
        Calculate specific pipe friction.

        Formula: R = Force / (π × D × L)

        Reference values:
        - < 0.5 t/m²: Good
        - 0.5-1.0 t/m²: Attention needed
        - > 1.0 t/m²: Critical

        Args:
            force_tonnes: Jacking force for section (tonnes)
            pipe_diameter_mm: Pipe outer diameter (mm)
            section_length_m: Length of pipe section (m)

        Returns:
            Dictionary with friction analysis
        """
        D = pipe_diameter_mm / 1000  # m

        # Pipe surface area
        surface_area = math.pi * D * section_length_m

        # Specific friction
        specific_friction = force_tonnes / surface_area

        # Status assessment
        if specific_friction < 0.5:
            status = "GOOD"
            recommendation = "Normal operation"
        elif specific_friction < 1.0:
            status = "ATTENTION"
            recommendation = "Monitor closely, consider additional lubrication"
        else:
            status = "CRITICAL"
            recommendation = "Reduce advance, increase lubrication, check alignment"

        return {
            'force_tonnes': force_tonnes,
            'pipe_diameter_mm': pipe_diameter_mm,
            'section_length_m': section_length_m,
            'surface_area_m2': surface_area,
            'specific_friction_t_m2': specific_friction,
            'status': status,
            'recommendation': recommendation,
        }

    def analyze_friction_sections(self,
                                  pipe_diameter_mm: float,
                                  machine_steering_force: float,
                                  expanders: List[ExpanderConfig],
                                  main_cylinder_force: float) -> List[Dict]:
        """
        Analyze specific friction for multiple sections with expanders.

        Args:
            pipe_diameter_mm: Pipe outer diameter (mm)
            machine_steering_force: Force at machine steering cylinders (tonnes)
            expanders: List of ExpanderConfig with positions and forces
            main_cylinder_force: Force at main jacking station (tonnes)

        Returns:
            List of friction analyses for each section
        """
        results = []

        # Sort expanders by position (closest to machine first)
        sorted_expanders = sorted(expanders, key=lambda x: x.position_m)

        # Section 1: First expander to machine
        if sorted_expanders:
            exp1 = sorted_expanders[0]
            force1 = exp1.force_tonnes - machine_steering_force
            result1 = self.calculate_specific_friction(
                force1, pipe_diameter_mm, exp1.position_m
            )
            result1['section'] = f"Section 1: {exp1.name or 'Expander 1'} to Machine"
            result1['net_force_tonnes'] = force1
            results.append(result1)

        # Middle sections between expanders
        for i in range(len(sorted_expanders) - 1):
            exp_front = sorted_expanders[i]
            exp_back = sorted_expanders[i + 1]
            section_length = exp_back.position_m - exp_front.position_m
            force = exp_back.force_tonnes

            result = self.calculate_specific_friction(
                force, pipe_diameter_mm, section_length
            )
            result['section'] = f"Section {i+2}: {exp_back.name or f'Expander {i+2}'} to {exp_front.name or f'Expander {i+1}'}"
            result['net_force_tonnes'] = force
            results.append(result)

        # Last section: Main cylinder to last expander
        if sorted_expanders:
            last_exp = sorted_expanders[-1]
            # Assume main cylinder is at the shaft
            # Section length would need total distance - last expander position
            result_main = self.calculate_specific_friction(
                main_cylinder_force, pipe_diameter_mm,
                last_exp.position_m  # This is approximate
            )
            result_main['section'] = f"Section {len(sorted_expanders)+1}: Main Cylinder to {last_exp.name or 'Last Expander'}"
            result_main['net_force_tonnes'] = main_cylinder_force
            results.append(result_main)

        return results

    def calculate_advance_time(self,
                               pipe_length_m: float,
                               stroke_length_mm: float,
                               advance_speed_cm_min: float,
                               retract_speed_cm_min: float,
                               changeover_time_min: float,
                               num_cylinder_sets: int = 3) -> Dict:
        """
        Calculate time to advance one pipe with expanders.

        Args:
            pipe_length_m: Pipe length (m)
            stroke_length_mm: Cylinder stroke length (mm)
            advance_speed_cm_min: Speed during advance (cm/min)
            retract_speed_cm_min: Speed during cylinder retraction (cm/min)
            changeover_time_min: Time to switch between cylinders (min)
            num_cylinder_sets: Number of cylinder sets (machine + expanders)

        Returns:
            Dictionary with time calculations
        """
        stroke_cm = stroke_length_mm / 10
        pipe_length_cm = pipe_length_m * 100

        # Number of strokes needed
        strokes_per_pipe = math.ceil(pipe_length_cm / stroke_cm)

        # Time for each stroke cycle
        advance_time = stroke_cm / advance_speed_cm_min
        retract_time = stroke_cm / retract_speed_cm_min
        total_changeovers = num_cylinder_sets - 1  # Changeovers between sets

        time_per_stroke = (advance_time +
                          (retract_time * (num_cylinder_sets - 1)) +
                          (changeover_time_min * total_changeovers))

        # Total time per pipe
        total_time_min = strokes_per_pipe * time_per_stroke

        return {
            'pipe_length_m': pipe_length_m,
            'stroke_length_mm': stroke_length_mm,
            'strokes_per_pipe': strokes_per_pipe,
            'advance_speed_cm_min': advance_speed_cm_min,
            'retract_speed_cm_min': retract_speed_cm_min,
            'changeover_time_min': changeover_time_min,
            'num_cylinder_sets': num_cylinder_sets,
            'time_per_stroke_min': time_per_stroke,
            'total_time_per_pipe_min': total_time_min,
            'pipes_per_hour': 60 / total_time_min if total_time_min > 0 else 0,
        }

    def calculate_slope(self,
                        distance_m: float,
                        slope_percent: float) -> Dict:
        """
        Calculate height difference from distance and slope.

        Args:
            distance_m: Horizontal distance (m)
            slope_percent: Slope in percent (e.g., 1.2 for 1.2%)

        Returns:
            Dictionary with slope calculations
        """
        # Height difference
        height_diff_m = distance_m * slope_percent / 100
        height_diff_cm = height_diff_m * 100
        height_diff_mm = height_diff_m * 1000

        # Slope in other units
        slope_permille = slope_percent * 10  # ‰ = mm/m
        slope_ratio = f"1:{100/slope_percent:.0f}" if slope_percent > 0 else "0"

        return {
            'distance_m': distance_m,
            'slope_percent': slope_percent,
            'slope_permille': slope_permille,
            'slope_ratio': slope_ratio,
            'height_difference_m': height_diff_m,
            'height_difference_cm': height_diff_cm,
            'height_difference_mm': height_diff_mm,
            'direction': 'upward' if slope_percent > 0 else 'downward' if slope_percent < 0 else 'level',
        }

    def calculate_penetration_rate(self,
                                   advance_speed_cm_min: float,
                                   cutter_rpm: float) -> Dict:
        """
        Calculate penetration per rotation.

        Formula: P = V / RPM (cm per rotation)

        Args:
            advance_speed_cm_min: Advance speed (cm/min)
            cutter_rpm: Cutterhead rotation speed (rpm)

        Returns:
            Dictionary with penetration calculations
        """
        if cutter_rpm <= 0:
            penetration_cm = 0
            penetration_mm = 0
        else:
            penetration_cm = advance_speed_cm_min / cutter_rpm
            penetration_mm = penetration_cm * 10

        return {
            'advance_speed_cm_min': advance_speed_cm_min,
            'cutter_rpm': cutter_rpm,
            'penetration_cm_per_rev': penetration_cm,
            'penetration_mm_per_rev': penetration_mm,
            'note': 'Higher penetration = more cutting per rotation',
        }

    def calculate_ground_content_from_density(self,
                                              slurry_density_kg_L: float,
                                              base_fluid_density_kg_L: float = 1.0) -> Dict:
        """
        Calculate ground material content from slurry density.

        Args:
            slurry_density_kg_L: Measured slurry density (kg/L)
            base_fluid_density_kg_L: Base fluid density (kg/L), default 1.0 for water

        Returns:
            Dictionary with ground content calculations
        """
        # Density difference = ground content
        density_diff = slurry_density_kg_L - base_fluid_density_kg_L

        # Convert to kg per m³
        ground_content_kg_m3 = density_diff * 1000
        ground_content_t_m3 = density_diff

        return {
            'slurry_density_kg_L': slurry_density_kg_L,
            'base_fluid_density_kg_L': base_fluid_density_kg_L,
            'density_difference_kg_L': density_diff,
            'ground_content_kg_m3': ground_content_kg_m3,
            'ground_content_t_m3': ground_content_t_m3,
        }

    def calculate_ground_from_density_change(self,
                                             initial_density_kg_L: float,
                                             final_density_kg_L: float,
                                             slurry_volume_m3: float) -> Dict:
        """
        Calculate ground entering slurry from density change.

        Args:
            initial_density_kg_L: Density at start of pipe advance (kg/L)
            final_density_kg_L: Density at end of pipe advance (kg/L)
            slurry_volume_m3: Volume of slurry in system (m³)

        Returns:
            Dictionary with ground material calculations
        """
        density_change = final_density_kg_L - initial_density_kg_L
        ground_content_kg_m3 = density_change * 1000

        # Total ground that entered
        total_ground_kg = ground_content_kg_m3 * slurry_volume_m3

        return {
            'initial_density_kg_L': initial_density_kg_L,
            'final_density_kg_L': final_density_kg_L,
            'density_change_kg_L': density_change,
            'ground_content_kg_m3': ground_content_kg_m3,
            'slurry_volume_m3': slurry_volume_m3,
            'total_ground_entered_kg': total_ground_kg,
        }

    def calculate_settlement_from_overcut(self,
                                          overcut_volume_m3_per_m: float,
                                          cutter_diameter_mm: float) -> Dict:
        """
        Calculate potential settlement from overcut volume.

        Two methods:
        1. Pure overcut = overcut per side
        2. Distributed = overcut volume / area above pipe

        Args:
            overcut_volume_m3_per_m: Overcut volume per running meter (m³/lfm)
            cutter_diameter_mm: Cutterhead diameter (mm)

        Returns:
            Dictionary with settlement calculations
        """
        D = cutter_diameter_mm / 1000  # m

        # Method 1: Pure overcut per side (worst case localized)
        # This would be (D_cutter - D_pipe) / 2, but we calculate from volume
        # Overcut area = V / L, solve for annular width
        overcut_area = overcut_volume_m3_per_m  # m² (for 1m length)

        # Method 2: Distributed over area above pipe
        # Area above pipe ≈ D × 1m (simplified rectangular area)
        area_above_pipe = D * 1.0  # m² per running meter
        settlement_distributed_m = overcut_volume_m3_per_m / area_above_pipe
        settlement_distributed_cm = settlement_distributed_m * 100

        return {
            'overcut_volume_m3_per_m': overcut_volume_m3_per_m,
            'cutter_diameter_mm': cutter_diameter_mm,
            'area_above_pipe_m2': area_above_pipe,
            'settlement_distributed_m': settlement_distributed_m,
            'settlement_distributed_cm': settlement_distributed_cm,
            'note': 'Worst case - assumes all overcut volume causes settlement',
        }

    def calculate_curve_radius_from_joints(self,
                                           joint_peak_mm: float,
                                           joint_right_mm: float,
                                           joint_invert_mm: float,
                                           joint_left_mm: float,
                                           pipe_diameter_mm: float,
                                           pipe_length_m: float) -> Dict:
        """
        Calculate pipe angle and curve radius from joint gap measurements.

        Args:
            joint_peak_mm: Joint gap at peak/crown (mm)
            joint_right_mm: Joint gap at right abutment (mm)
            joint_invert_mm: Joint gap at invert (mm)
            joint_left_mm: Joint gap at left abutment (mm)
            pipe_diameter_mm: Pipe outer diameter (mm)
            pipe_length_m: Pipe length (m)

        Returns:
            Dictionary with curve analysis
        """
        D = pipe_diameter_mm / 1000  # m

        # Horizontal gap (left-right difference)
        horizontal_gap_mm = joint_left_mm - joint_right_mm

        # Vertical gap (peak-invert difference)
        vertical_gap_mm = joint_peak_mm - joint_invert_mm

        # Horizontal angle (curve left/right)
        if horizontal_gap_mm != 0:
            angle_h_rad = math.atan(abs(horizontal_gap_mm) / 1000 / D)
            angle_h_deg = math.degrees(angle_h_rad)
            direction_h = "RIGHT" if horizontal_gap_mm > 0 else "LEFT"
        else:
            angle_h_deg = 0
            direction_h = "STRAIGHT"

        # Vertical angle (up/down)
        if vertical_gap_mm != 0:
            angle_v_rad = math.atan(abs(vertical_gap_mm) / 1000 / D)
            angle_v_deg = math.degrees(angle_v_rad)
            direction_v = "DOWN" if vertical_gap_mm > 0 else "UP"
        else:
            angle_v_deg = 0
            direction_v = "LEVEL"

        # Calculate curve radius (horizontal)
        if angle_h_deg > 0:
            # Number of pipes for full circle = 360 / angle - 1
            pipes_per_circle = (360 / angle_h_deg) - 1 if angle_h_deg > 0 else float('inf')
            # Circumference = pipes × pipe_length
            circumference = pipes_per_circle * pipe_length_m
            # Radius = circumference / (2π)
            curve_radius_m = circumference / (2 * math.pi)
        else:
            pipes_per_circle = float('inf')
            curve_radius_m = float('inf')

        return {
            'joint_peak_mm': joint_peak_mm,
            'joint_right_mm': joint_right_mm,
            'joint_invert_mm': joint_invert_mm,
            'joint_left_mm': joint_left_mm,
            'horizontal_gap_mm': horizontal_gap_mm,
            'vertical_gap_mm': vertical_gap_mm,
            'horizontal_angle_deg': angle_h_deg,
            'horizontal_direction': direction_h,
            'vertical_angle_deg': angle_v_deg,
            'vertical_direction': direction_v,
            'curve_radius_m': curve_radius_m,
            'pipes_per_full_circle': pipes_per_circle,
        }

    def calculate_sedimentation_bentonite(self,
                                          tank_volume_m3: float,
                                          bentonite_concentration_kg_m3: float = 30,
                                          renewals_per_day: int = 1) -> Dict:
        """
        Calculate bentonite for sedimentation tank renewal.

        Args:
            tank_volume_m3: Sedimentation tank volume (m³)
            bentonite_concentration_kg_m3: Bentonite per m³ suspension (kg/m³)
            renewals_per_day: Number of times suspension is renewed per day

        Returns:
            Dictionary with bentonite requirements
        """
        bentonite_per_renewal_kg = tank_volume_m3 * bentonite_concentration_kg_m3
        bentonite_per_day_kg = bentonite_per_renewal_kg * renewals_per_day

        return {
            'tank_volume_m3': tank_volume_m3,
            'bentonite_concentration_kg_m3': bentonite_concentration_kg_m3,
            'renewals_per_day': renewals_per_day,
            'bentonite_per_renewal_kg': bentonite_per_renewal_kg,
            'bentonite_per_day_kg': bentonite_per_day_kg,
        }

    def calculate_daily_bentonite(self,
                                  overcut_flow_m3_min: float,
                                  lubrication_volume_m3_per_pipe: float,
                                  pipe_length_m: float,
                                  pipes_per_day: int,
                                  bentonite_concentration_kg_m3: float = 35) -> Dict:
        """
        Calculate total daily bentonite requirement.

        Args:
            overcut_flow_m3_min: Bentonite flow rate for overcut (m³/min)
            lubrication_volume_m3_per_pipe: Volume per pipe for lubrication (m³)
            pipe_length_m: Pipe length (m)
            pipes_per_day: Number of pipes installed per day
            bentonite_concentration_kg_m3: Bentonite concentration (kg/m³)

        Returns:
            Dictionary with daily bentonite requirements
        """
        # Volume for overcut per pipe (assuming time calculation)
        # This is simplified - actual depends on advance time
        overcut_volume_per_pipe = overcut_flow_m3_min * pipe_length_m * 10  # Approximate

        # Total suspension volume per pipe
        total_suspension_per_pipe = overcut_volume_per_pipe + lubrication_volume_m3_per_pipe

        # Daily totals
        daily_suspension_m3 = total_suspension_per_pipe * pipes_per_day
        daily_bentonite_kg = daily_suspension_m3 * bentonite_concentration_kg_m3

        return {
            'overcut_flow_m3_min': overcut_flow_m3_min,
            'lubrication_volume_m3_per_pipe': lubrication_volume_m3_per_pipe,
            'total_suspension_per_pipe_m3': total_suspension_per_pipe,
            'pipes_per_day': pipes_per_day,
            'daily_suspension_m3': daily_suspension_m3,
            'bentonite_concentration_kg_m3': bentonite_concentration_kg_m3,
            'daily_bentonite_kg': daily_bentonite_kg,
        }


def main():
    """Demonstrate calculator usage"""
    calc = PipejackingCalculator()
    
    print("="*80)
    print("HITCHHIKER'S GUIDE TO PIPEJACKING - CALCULATOR")
    print("="*80)
    
    # Example 1: Main Jacking Station
    print("\n" + "="*80)
    print("EXAMPLE 1: Main Jacking Station")
    print("="*80)
    
    main_station = JackingStationParams(
        num_cylinders=4,
        cylinder_diameter_mm=285,
        oil_flow_lpm=60,
        max_pressure_bar=500
    )
    
    main_results = calc.calculate_jacking_force(main_station)
    
    print(f"Input Parameters:")
    print(f"  Number of cylinders: {main_station.num_cylinders}")
    print(f"  Cylinder diameter: {main_station.cylinder_diameter_mm} mm")
    print(f"  Oil flow: {main_station.oil_flow_lpm} L/min")
    print(f"  Max pressure: {main_station.max_pressure_bar} bar")
    
    print(f"\nCalculated Results:")
    print(f"  Combined full bore area: {main_results.combined_area_cm2:.4f} cm²")
    print(f"  Total force: {main_results.total_force_tonnes:.5f} tonnes")
    print(f"  Max speed: {main_results.max_speed_cm_per_min:.2f} cm/min")
    print(f"  Time per pipe (3m): {main_results.time_per_pipe_minutes:.2f} minutes")
    
    # Example 2: Interjack Station
    print("\n" + "="*80)
    print("EXAMPLE 2: Interjack Station")
    print("="*80)
    
    interjack_station = JackingStationParams(
        num_cylinders=16,
        cylinder_diameter_mm=140,
        oil_flow_lpm=60,
        max_pressure_bar=500
    )
    
    interjack_results = calc.calculate_jacking_force(interjack_station)
    
    print(f"Input Parameters:")
    print(f"  Number of cylinders: {interjack_station.num_cylinders}")
    print(f"  Cylinder diameter: {interjack_station.cylinder_diameter_mm} mm")
    
    print(f"\nCalculated Results:")
    print(f"  Combined full bore area: {interjack_results.combined_area_cm2:.4f} cm²")
    print(f"  Total force: {interjack_results.total_force_tonnes:.3f} tonnes")
    
    # Example 3: Slurry Flow Rates
    print("\n" + "="*80)
    print("EXAMPLE 3: Slurry Flow Rates (150mm pipe)")
    print("="*80)
    
    slurry_params = SlurryParams(pipe_diameter_mm=150)
    flow_table = calc.generate_slurry_flow_table(slurry_params)
    
    print(f"\n{'Velocity (m/s)':<15} {'Flow Rate (m³/hr)':<20} {'Flow Rate (L/min)':<20}")
    print("-"*60)
    for flow in flow_table:
        print(f"{flow['velocity_mps']:<15.2f} {flow['flow_rate_m3_per_hr']:<20.2f} {flow['flow_rate_l_per_min']:<20.2f}")
    
    # Example 4: Power Requirements
    print("\n" + "="*80)
    print("EXAMPLE 4: Power Requirements")
    print("="*80)
    
    motor_powers = [30, 110, 30, 30, 30, 250, 175, 0, 0, 0]  # kW
    power_params = PowerParams(power_factor=0.8, voltage=400)
    
    power_results = calc.calculate_power_requirements(motor_powers, power_params)
    
    print(f"Motor Powers: {motor_powers} kW")
    print(f"\nCalculated Results:")
    print(f"  Total power: {power_results['total_kw']:.0f} kW")
    print(f"  Total power: {power_results['total_kva']:.2f} kVA")
    print(f"  Generator size (80%): {power_results['generator_size_kva']:.2f} kVA")
    
    # Example 5: Cable Sizing
    print("\n" + "="*80)
    print("EXAMPLE 5: Cable Sizing")
    print("="*80)
    
    # Calculate amperage for different loads
    test_loads = [
        (655, 400, "Container M"),
        (50, 400, "Separation Plant"),
        (9.5, 415, "Other equipment")
    ]
    
    for kva, voltage, name in test_loads:
        amps = calc.calculate_cable_amperage(kva, voltage)
        cable_info = calc.get_cable_size(amps)
        
        print(f"\n{name}:")
        print(f"  Power: {kva} kVA at {voltage}V")
        print(f"  Amperage: {amps:.2f} Amps")
        print(f"  Recommended cable: {cable_info['recommended_size_mm2']} mm²")
        print(f"  Cable rating: {cable_info['cable_rating_amps']} Amps")
    
    print("\n" + "="*80)
    print("Calculator demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()

