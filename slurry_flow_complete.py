"""
Complete Slurry Flow Calculator for Microtunneling
====================================================
Calculates slurry flow requirements based on:
- Machine excavation rate
- Overcut volume
- Soil bulking factors
- Transport velocity requirements
- Pipe sizing verification

Based on industry standards and Lutz Henke's Hitchhiker's Guide
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class MachineParams:
    """Microtunneling machine parameters"""
    cutter_diameter_mm: float      # Cutter head diameter
    pipe_od_mm: float              # Pipe outer diameter
    advance_rate_mm_min: float     # Advance rate


@dataclass
class SlurryPipeParams:
    """Slurry pipe parameters"""
    feed_pipe_diameter_mm: float   # Feed line diameter
    return_pipe_diameter_mm: float # Return/discharge line diameter


@dataclass
class SoilParams:
    """Soil parameters"""
    soil_type: str                 # Sand, Clay, Gravel, Mixed
    bulking_factor: float          # Volume increase when excavated (1.1-1.4)
    max_concentration: float       # Maximum Cv for transport (0.15-0.30)
    min_velocity: float            # Minimum transport velocity (m/s)


# Soil type presets
SOIL_PRESETS = {
    'sand': SoilParams(
        soil_type='Sand',
        bulking_factor=1.15,
        max_concentration=0.25,
        min_velocity=3.0
    ),
    'clay': SoilParams(
        soil_type='Clay',
        bulking_factor=1.25,
        max_concentration=0.20,
        min_velocity=2.5
    ),
    'gravel': SoilParams(
        soil_type='Gravel',
        bulking_factor=1.10,
        max_concentration=0.15,
        min_velocity=3.5
    ),
    'mixed': SoilParams(
        soil_type='Mixed Ground',
        bulking_factor=1.20,
        max_concentration=0.20,
        min_velocity=3.0
    ),
    'rock': SoilParams(
        soil_type='Soft Rock',
        bulking_factor=1.40,
        max_concentration=0.15,
        min_velocity=4.0
    ),
}


@dataclass
class ParticleParams:
    """Particle characteristics for settling/transport calculations"""
    d50_mm: float              # Median particle size (mm)
    d85_mm: float = None       # 85th percentile particle size (mm)
    particle_sg: float = 2.65  # Specific gravity of particles (silica = 2.65)
    shape_factor: float = 0.7  # Sphericity (1.0 = perfect sphere, 0.5-0.8 typical)


@dataclass
class SlurryCircuitParams:
    """Parameters for slurry circuit mass balance"""
    Q_in_m3_hr: float         # Feed line flow rate (m³/hr)
    Q_out_m3_hr: float        # Return line flow rate (m³/hr)
    rho_in: float             # Feed slurry density (kg/m³ or g/cm³ * 1000)
    rho_out: float            # Return slurry density (kg/m³)
    rho_solids: float = 2650  # Solid particle density (kg/m³)
    rho_water: float = 1000   # Carrier fluid density (kg/m³)


# Bentonite injection rates by ground type (L/m of tunnel per m of diameter)
BENTONITE_INJECTION = {
    'clay': {'rate_L_per_m_per_mD': 15, 'description': 'Low permeability'},
    'silt': {'rate_L_per_m_per_mD': 25, 'description': 'Moderate permeability'},
    'sand': {'rate_L_per_m_per_mD': 40, 'description': 'High permeability'},
    'gravel': {'rate_L_per_m_per_mD': 60, 'description': 'Very high permeability'},
    'mixed': {'rate_L_per_m_per_mD': 35, 'description': 'Average conditions'},
    'rock': {'rate_L_per_m_per_mD': 20, 'description': 'Fractured rock'},
}


class SlurryFlowCalculator:
    """
    Complete slurry flow calculator for microtunneling operations
    """

    def __init__(self):
        self.results = {}

    # =========================================================================
    # CRITICAL VELOCITY CALCULATIONS
    # =========================================================================

    def calculate_settling_velocity(self, particle: ParticleParams,
                                    fluid_density: float = 1000,
                                    fluid_viscosity: float = 0.001) -> Dict:
        """
        Calculate particle settling velocity using appropriate method.

        For small particles (Re < 1): Stokes' Law
        For larger particles: Drag equation with empirical correlation

        Args:
            particle: ParticleParams with particle characteristics
            fluid_density: Carrier fluid density (kg/m³), default water = 1000
            fluid_viscosity: Dynamic viscosity (Pa·s), default water = 0.001

        Returns:
            Dictionary with settling velocity and regime information
        """
        g = 9.81  # gravity (m/s²)
        d_m = particle.d50_mm / 1000  # Convert mm to m
        rho_s = particle.particle_sg * 1000  # Particle density (kg/m³)
        rho_f = fluid_density
        mu = fluid_viscosity

        # Stokes settling velocity (valid for Re < 1)
        v_stokes = (g * d_m**2 * (rho_s - rho_f)) / (18 * mu)

        # Calculate Reynolds number for Stokes velocity
        Re_stokes = (rho_f * v_stokes * d_m) / mu

        # Determine flow regime and appropriate formula
        if Re_stokes < 1:
            # Stokes regime (laminar)
            v_settling = v_stokes
            regime = "Stokes (laminar)"
            drag_coeff = 24 / Re_stokes if Re_stokes > 0 else float('inf')
        elif Re_stokes < 1000:
            # Intermediate regime - use Schiller-Naumann correlation
            # Iterative solution for drag coefficient
            v_settling = self._solve_intermediate_settling(
                d_m, rho_s, rho_f, mu, g
            )
            Re_actual = (rho_f * v_settling * d_m) / mu
            drag_coeff = (24 / Re_actual) * (1 + 0.15 * Re_actual**0.687)
            regime = "Intermediate (transitional)"
        else:
            # Newton regime (turbulent) - Cd ≈ 0.44
            drag_coeff = 0.44
            v_settling = math.sqrt((4 * g * d_m * (rho_s - rho_f)) /
                                   (3 * drag_coeff * rho_f))
            regime = "Newton (turbulent)"

        # Apply shape factor correction
        v_settling *= particle.shape_factor

        # Recalculate actual Reynolds number
        Re_actual = (rho_f * v_settling * d_m) / mu

        return {
            'particle_diameter_mm': particle.d50_mm,
            'particle_sg': particle.particle_sg,
            'settling_velocity_mps': v_settling,
            'settling_velocity_mm_s': v_settling * 1000,
            'reynolds_number': Re_actual,
            'flow_regime': regime,
            'drag_coefficient': drag_coeff,
            'shape_factor_applied': particle.shape_factor,
        }

    def _solve_intermediate_settling(self, d: float, rho_s: float,
                                     rho_f: float, mu: float, g: float) -> float:
        """Iterative solution for intermediate regime settling velocity"""
        # Initial guess using Stokes
        v = (g * d**2 * (rho_s - rho_f)) / (18 * mu)

        for _ in range(50):  # Max iterations
            Re = (rho_f * v * d) / mu
            if Re < 0.1:
                Re = 0.1
            Cd = (24 / Re) * (1 + 0.15 * Re**0.687)
            v_new = math.sqrt((4 * g * d * (rho_s - rho_f)) / (3 * Cd * rho_f))
            if abs(v_new - v) / v < 0.001:  # Convergence check
                break
            v = v_new
        return v

    def calculate_critical_velocity(self,
                                    pipe_diameter_mm: float,
                                    particle: ParticleParams,
                                    Cv: float,
                                    fluid_density: float = 1000) -> Dict:
        """
        Calculate critical deposition velocity using Durand-Condolios formula.

        This is the minimum velocity to prevent particle settling in the pipe.

        Durand Formula: V_c = F_L × √(2gD(S-1))

        Where:
            F_L = Durand factor (depends on particle size and concentration)
            g = gravity (9.81 m/s²)
            D = pipe internal diameter (m)
            S = particle specific gravity (relative to fluid)

        Args:
            pipe_diameter_mm: Internal pipe diameter (mm)
            particle: ParticleParams with particle characteristics
            Cv: Volume concentration (0-1, not percentage)
            fluid_density: Carrier fluid density (kg/m³)

        Returns:
            Dictionary with critical velocity calculations
        """
        g = 9.81
        D = pipe_diameter_mm / 1000  # Convert to meters
        S = particle.particle_sg
        d = particle.d50_mm / 1000   # Particle diameter in meters

        # Calculate Durand factor F_L
        # F_L depends on particle size ratio (d/D) and concentration
        # Empirical correlation from Durand (1953)

        # Particle size ratio
        d_over_D = d / D

        # F_L calculation (simplified Durand correlation)
        # For fine particles (d < 0.5mm): F_L increases with concentration
        # For coarse particles (d > 2mm): F_L decreases with size

        if particle.d50_mm < 0.5:
            # Fine particles - lower critical velocity
            F_L_base = 0.9 + 0.3 * math.sqrt(Cv * 100)  # 0.9-1.5 typical
        elif particle.d50_mm < 2.0:
            # Medium particles - highest critical velocity
            F_L_base = 1.3 + 0.2 * math.sqrt(Cv * 100)  # 1.3-1.8 typical
        else:
            # Coarse particles - F_L decreases
            F_L_base = 1.1 + 0.15 * math.sqrt(Cv * 100)

        # Cap F_L at reasonable values (0.8 to 2.0)
        F_L = max(0.8, min(2.0, F_L_base))

        # Durand critical velocity
        V_critical = F_L * math.sqrt(2 * g * D * (S - 1))

        # Also calculate using Wasp method for comparison
        # Wasp: V_c = 3.8 × D^0.5 × (d/D)^0.167 × [(S-1)×g]^0.333
        V_wasp = 3.8 * (D**0.5) * (d_over_D**0.167) * ((S - 1) * g)**0.333

        # Recommended velocity (add 20% safety margin)
        V_recommended = max(V_critical, V_wasp) * 1.2

        return {
            'pipe_diameter_mm': pipe_diameter_mm,
            'particle_d50_mm': particle.d50_mm,
            'particle_sg': particle.particle_sg,
            'volume_concentration_Cv': Cv,
            'durand_factor_FL': F_L,
            'critical_velocity_durand_mps': V_critical,
            'critical_velocity_wasp_mps': V_wasp,
            'recommended_minimum_mps': V_recommended,
            'description': f"Minimum velocity to prevent settling of {particle.d50_mm}mm particles"
        }

    # =========================================================================
    # EXCAVATION MASS BALANCE
    # =========================================================================

    def calculate_excavation_mass_balance(self,
                                          circuit: SlurryCircuitParams,
                                          time_hours: float = 1.0) -> Dict:
        """
        Calculate excavated volume and mass using slurry circuit mass balance.

        Based on Duhme et al. (2016): Mass_in + Mass_excavated = Mass_out

        Args:
            circuit: SlurryCircuitParams with flow rates and densities
            time_hours: Time period for calculation (default 1 hour)

        Returns:
            Dictionary with excavation quantities
        """
        Q_in = circuit.Q_in_m3_hr
        Q_out = circuit.Q_out_m3_hr
        rho_in = circuit.rho_in
        rho_out = circuit.rho_out
        rho_s = circuit.rho_solids
        rho_w = circuit.rho_water

        # Method 1: Volume difference
        # V_excavated = Q_out - Q_in (if densities are similar)
        V_diff_m3_hr = Q_out - Q_in

        # Method 2: Mass balance (more accurate)
        # Mass_in = Q_in × ρ_in
        # Mass_out = Q_out × ρ_out
        # Mass_excavated = Mass_out - Mass_in
        mass_in_kg_hr = Q_in * rho_in
        mass_out_kg_hr = Q_out * rho_out
        mass_excavated_kg_hr = mass_out_kg_hr - mass_in_kg_hr

        # Method 3: Solids content calculation
        # Calculate solids concentration from density
        # Cv = (ρ_slurry - ρ_water) / (ρ_solids - ρ_water)
        Cv_in = max(0, (rho_in - rho_w) / (rho_s - rho_w))
        Cv_out = max(0, (rho_out - rho_w) / (rho_s - rho_w))

        # Solids volume flow rates
        solids_in_m3_hr = Q_in * Cv_in
        solids_out_m3_hr = Q_out * Cv_out

        # Net excavated solids volume
        solids_excavated_m3_hr = solids_out_m3_hr - solids_in_m3_hr

        # Dry mass of excavated material
        dry_mass_kg_hr = solids_excavated_m3_hr * rho_s

        # Total values for time period
        total_volume_m3 = solids_excavated_m3_hr * time_hours
        total_mass_kg = dry_mass_kg_hr * time_hours
        total_mass_tonnes = total_mass_kg / 1000

        return {
            'time_period_hours': time_hours,
            'flow_rate_in_m3_hr': Q_in,
            'flow_rate_out_m3_hr': Q_out,
            'density_in_kg_m3': rho_in,
            'density_out_kg_m3': rho_out,
            'concentration_Cv_in': Cv_in,
            'concentration_Cv_out': Cv_out,
            'solids_in_m3_hr': solids_in_m3_hr,
            'solids_out_m3_hr': solids_out_m3_hr,
            'excavated_volume_m3_hr': solids_excavated_m3_hr,
            'excavated_dry_mass_kg_hr': dry_mass_kg_hr,
            'total_excavated_volume_m3': total_volume_m3,
            'total_excavated_mass_tonnes': total_mass_tonnes,
            'mass_balance_check_kg_hr': mass_in_kg_hr + dry_mass_kg_hr - mass_out_kg_hr,
        }

    def check_overexcavation(self,
                             measured_volume_m3: float,
                             theoretical_volume_m3: float,
                             tolerance_percent: float = 5.0) -> Dict:
        """
        Check for over-excavation by comparing measured vs theoretical volume.

        Args:
            measured_volume_m3: Volume calculated from mass balance
            theoretical_volume_m3: Volume from advance rate × cutter area
            tolerance_percent: Acceptable variance (default 5%)

        Returns:
            Dictionary with over-excavation analysis
        """
        difference_m3 = measured_volume_m3 - theoretical_volume_m3
        difference_percent = (difference_m3 / theoretical_volume_m3) * 100 if theoretical_volume_m3 > 0 else 0

        if difference_percent > tolerance_percent:
            status = "OVER-EXCAVATION WARNING"
            recommendation = "Check for ground loss, voids, or groundwater ingress"
        elif difference_percent < -tolerance_percent:
            status = "UNDER-EXCAVATION WARNING"
            recommendation = "Check for cutter head blockage or sensor calibration"
        else:
            status = "NORMAL"
            recommendation = "Excavation within acceptable tolerance"

        return {
            'measured_volume_m3': measured_volume_m3,
            'theoretical_volume_m3': theoretical_volume_m3,
            'difference_m3': difference_m3,
            'difference_percent': difference_percent,
            'tolerance_percent': tolerance_percent,
            'status': status,
            'recommendation': recommendation,
        }

    # =========================================================================
    # BENTONITE INJECTION CALCULATIONS
    # =========================================================================

    def calculate_bentonite_injection(self,
                                      pipe_od_mm: float,
                                      tunnel_length_m: float,
                                      ground_type: str,
                                      advance_rate_m_hr: float = 1.0) -> Dict:
        """
        Calculate bentonite injection requirements for pipe lubrication.

        Based on Praetorius & Schößer (2017) Bentonite Handbook.

        Args:
            pipe_od_mm: Pipe outer diameter (mm)
            tunnel_length_m: Total tunnel length (m)
            ground_type: Type of ground (clay, silt, sand, gravel, mixed, rock)
            advance_rate_m_hr: Expected advance rate (m/hr)

        Returns:
            Dictionary with bentonite injection requirements
        """
        pipe_od_m = pipe_od_mm / 1000

        # Get injection rate for ground type
        ground_key = ground_type.lower()
        if ground_key not in BENTONITE_INJECTION:
            ground_key = 'mixed'  # Default

        injection_data = BENTONITE_INJECTION[ground_key]
        base_rate = injection_data['rate_L_per_m_per_mD']

        # Injection rate scales with pipe diameter
        injection_rate_L_per_m = base_rate * pipe_od_m

        # Total bentonite required for project
        total_bentonite_L = injection_rate_L_per_m * tunnel_length_m

        # Pumping rate based on advance rate
        pumping_rate_L_hr = injection_rate_L_per_m * advance_rate_m_hr
        pumping_rate_L_min = pumping_rate_L_hr / 60

        # Bentonite mix preparation (typical 5% mix)
        bentonite_powder_kg = total_bentonite_L * 0.05  # 5% concentration
        water_required_L = total_bentonite_L * 0.95

        return {
            'ground_type': ground_type,
            'ground_description': injection_data['description'],
            'pipe_diameter_mm': pipe_od_mm,
            'tunnel_length_m': tunnel_length_m,
            'base_rate_L_per_m_per_mD': base_rate,
            'injection_rate_L_per_m': injection_rate_L_per_m,
            'total_bentonite_mix_L': total_bentonite_L,
            'total_bentonite_mix_m3': total_bentonite_L / 1000,
            'bentonite_powder_kg': bentonite_powder_kg,
            'water_required_L': water_required_L,
            'pumping_rate_L_hr': pumping_rate_L_hr,
            'pumping_rate_L_min': pumping_rate_L_min,
            'advance_rate_m_hr': advance_rate_m_hr,
        }

    # =========================================================================
    # SLURRY HEAD LOSS WITH CORRECTIONS
    # =========================================================================

    def calculate_slurry_head_loss(self,
                                   pipe_diameter_mm: float,
                                   pipe_length_m: float,
                                   flow_rate_m3_hr: float,
                                   slurry_density_kg_m3: float,
                                   particle: ParticleParams,
                                   Cv: float) -> Dict:
        """
        Calculate head loss in slurry pipeline with corrections for solids.

        Uses Durand-Condolios method with concentration and particle corrections.

        Head loss multiplier: φ = 1 + K × Cv × [(gD(S-1))/V²]^1.5

        Args:
            pipe_diameter_mm: Internal pipe diameter (mm)
            pipe_length_m: Pipe length (m)
            flow_rate_m3_hr: Volumetric flow rate (m³/hr)
            slurry_density_kg_m3: Slurry density (kg/m³)
            particle: ParticleParams with particle characteristics
            Cv: Volume concentration (0-1)

        Returns:
            Dictionary with head loss calculations
        """
        g = 9.81
        D = pipe_diameter_mm / 1000  # meters
        L = pipe_length_m
        Q = flow_rate_m3_hr / 3600   # m³/s
        rho = slurry_density_kg_m3
        S = particle.particle_sg

        # Pipe area and velocity
        A = math.pi * (D / 2)**2
        V = Q / A  # velocity (m/s)

        # Reynolds number (using water viscosity as approximation)
        mu = 0.001  # Pa·s
        Re = (rho * V * D) / mu

        # Darcy friction factor (Swamee-Jain for turbulent flow)
        if Re > 4000:
            # Assume relative roughness e/D = 0.0001 for steel pipe
            e_D = 0.0001
            f_water = 0.25 / (math.log10(e_D/3.7 + 5.74/Re**0.9))**2
        else:
            f_water = 64 / Re  # Laminar

        # Water-only head loss (Darcy-Weisbach)
        h_water = f_water * (L / D) * (V**2 / (2 * g))

        # Durand-Condolios correction for slurry
        # φ = 1 + K × Cv × [(gD(S-1))/V²]^1.5
        # K depends on particle size (81 for fine, 150 for coarse)

        if particle.d50_mm < 0.5:
            K = 81   # Fine particles
        elif particle.d50_mm < 2.0:
            K = 121  # Medium particles
        else:
            K = 150  # Coarse particles

        # Froude number based term
        Fr_term = (g * D * (S - 1)) / V**2

        # Slurry correction factor
        if V > 0.1:  # Avoid division issues
            phi = 1 + K * Cv * (Fr_term**1.5)
        else:
            phi = 1 + K * Cv

        # Cap phi at reasonable value
        phi = min(phi, 5.0)

        # Slurry head loss
        h_slurry = h_water * phi

        # Pressure loss
        pressure_loss_Pa = rho * g * h_slurry
        pressure_loss_bar = pressure_loss_Pa / 100000

        return {
            'pipe_diameter_mm': pipe_diameter_mm,
            'pipe_length_m': pipe_length_m,
            'flow_rate_m3_hr': flow_rate_m3_hr,
            'velocity_mps': V,
            'reynolds_number': Re,
            'friction_factor_water': f_water,
            'head_loss_water_m': h_water,
            'durand_K_factor': K,
            'slurry_correction_phi': phi,
            'head_loss_slurry_m': h_slurry,
            'head_loss_increase_percent': (phi - 1) * 100,
            'pressure_loss_bar': pressure_loss_bar,
            'slurry_density_kg_m3': slurry_density_kg_m3,
            'volume_concentration_Cv': Cv,
        }

    def calculate_excavation(self, machine: MachineParams) -> Dict:
        """
        Calculate excavation volumes and rates

        Args:
            machine: MachineParams with machine specifications

        Returns:
            Dictionary with excavation calculations
        """
        # Cutter head area (excavation area)
        cutter_radius_m = (machine.cutter_diameter_mm / 2) / 1000
        cutter_area_m2 = math.pi * cutter_radius_m ** 2

        # Pipe area (for reference)
        pipe_radius_m = (machine.pipe_od_mm / 2) / 1000
        pipe_area_m2 = math.pi * pipe_radius_m ** 2

        # Overcut area (annular space)
        overcut_area_m2 = cutter_area_m2 - pipe_area_m2
        overcut_mm = (machine.cutter_diameter_mm - machine.pipe_od_mm) / 2

        # Advance rate conversion
        advance_rate_m_min = machine.advance_rate_mm_min / 1000
        advance_rate_m_hr = advance_rate_m_min * 60

        # Excavation volume rate
        excavation_m3_min = cutter_area_m2 * advance_rate_m_min
        excavation_m3_hr = excavation_m3_min * 60

        # Overcut volume rate (for bentonite/lubrication)
        overcut_m3_min = overcut_area_m2 * advance_rate_m_min
        overcut_m3_hr = overcut_m3_min * 60

        return {
            'cutter_diameter_mm': machine.cutter_diameter_mm,
            'cutter_area_m2': cutter_area_m2,
            'pipe_od_mm': machine.pipe_od_mm,
            'pipe_area_m2': pipe_area_m2,
            'overcut_mm': overcut_mm,
            'overcut_area_m2': overcut_area_m2,
            'advance_rate_mm_min': machine.advance_rate_mm_min,
            'advance_rate_m_hr': advance_rate_m_hr,
            'excavation_m3_min': excavation_m3_min,
            'excavation_m3_hr': excavation_m3_hr,
            'overcut_m3_hr': overcut_m3_hr,
        }

    def calculate_slurry_requirements(self,
                                      excavation: Dict,
                                      soil: SoilParams) -> Dict:
        """
        Calculate required slurry flow based on excavation rate

        Args:
            excavation: Results from calculate_excavation()
            soil: SoilParams with soil properties

        Returns:
            Dictionary with slurry flow requirements
        """
        # In-situ excavation rate
        insitu_m3_hr = excavation['excavation_m3_hr']

        # Bulked volume (soil expands when excavated)
        bulked_m3_hr = insitu_m3_hr * soil.bulking_factor

        # Required slurry flow to transport excavated material
        # Q_slurry = V_bulked / Cv
        # Where Cv is volume concentration
        required_flow_m3_hr = bulked_m3_hr / soil.max_concentration

        # Convert to L/min
        required_flow_L_min = required_flow_m3_hr * 1000 / 60

        # Recommended flow with safety factor (1.2-1.5)
        safety_factor = 1.3
        recommended_flow_m3_hr = required_flow_m3_hr * safety_factor
        recommended_flow_L_min = recommended_flow_m3_hr * 1000 / 60

        return {
            'soil_type': soil.soil_type,
            'bulking_factor': soil.bulking_factor,
            'max_concentration_Cv': soil.max_concentration,
            'min_velocity_mps': soil.min_velocity,
            'insitu_volume_m3_hr': insitu_m3_hr,
            'bulked_volume_m3_hr': bulked_m3_hr,
            'required_flow_m3_hr': required_flow_m3_hr,
            'required_flow_L_min': required_flow_L_min,
            'safety_factor': safety_factor,
            'recommended_flow_m3_hr': recommended_flow_m3_hr,
            'recommended_flow_L_min': recommended_flow_L_min,
        }

    def verify_pipe_sizing(self,
                          slurry_req: Dict,
                          slurry_pipe: SlurryPipeParams,
                          soil: SoilParams) -> Dict:
        """
        Verify slurry pipe sizing for required flow

        Args:
            slurry_req: Results from calculate_slurry_requirements()
            slurry_pipe: SlurryPipeParams with pipe sizes
            soil: SoilParams for velocity requirements

        Returns:
            Dictionary with pipe verification results
        """
        results = {}

        # Check both feed and return pipes
        for pipe_type, diameter_mm in [
            ('feed', slurry_pipe.feed_pipe_diameter_mm),
            ('return', slurry_pipe.return_pipe_diameter_mm)
        ]:
            # Pipe area
            radius_m = (diameter_mm / 2) / 1000
            area_m2 = math.pi * radius_m ** 2

            # Flow rate in m³/s
            flow_m3_s = slurry_req['recommended_flow_m3_hr'] / 3600

            # Velocity
            velocity_mps = flow_m3_s / area_m2

            # Check against minimum
            velocity_ok = velocity_mps >= soil.min_velocity
            velocity_status = "OK" if velocity_ok else "TOO LOW"

            # Maximum velocity check (typically 5.0 m/s)
            max_velocity = 5.0
            if velocity_mps > max_velocity:
                velocity_status = "TOO HIGH"
                velocity_ok = False

            # Calculate what diameter would be needed for optimal velocity
            optimal_velocity = (soil.min_velocity + max_velocity) / 2
            optimal_area = flow_m3_s / optimal_velocity
            optimal_diameter_mm = 2 * math.sqrt(optimal_area / math.pi) * 1000

            results[pipe_type] = {
                'diameter_mm': diameter_mm,
                'area_m2': area_m2,
                'velocity_mps': velocity_mps,
                'min_velocity_required': soil.min_velocity,
                'max_velocity_allowed': max_velocity,
                'velocity_status': velocity_status,
                'velocity_ok': velocity_ok,
                'optimal_diameter_mm': optimal_diameter_mm,
            }

        return results

    def calculate_pump_requirements(self,
                                   slurry_req: Dict,
                                   pipe_verification: Dict,
                                   pipe_length_m: float = 200) -> Dict:
        """
        Calculate pump requirements

        Args:
            slurry_req: Slurry requirements
            pipe_verification: Pipe verification results
            pipe_length_m: Total pipe length (default 200m)

        Returns:
            Dictionary with pump requirements
        """
        # Get return pipe velocity (highest resistance)
        return_vel = pipe_verification['return']['velocity_mps']
        return_dia = pipe_verification['return']['diameter_mm'] / 1000  # to m

        # Friction head loss (Darcy-Weisbach simplified)
        # Assuming slurry friction factor ~0.02-0.03
        friction_factor = 0.025
        g = 9.81  # gravity

        # Head loss = f × (L/D) × (v²/2g)
        head_loss_m = friction_factor * (pipe_length_m / return_dia) * (return_vel**2 / (2 * g))

        # Add static head (assume 10m typical shaft depth)
        static_head_m = 10
        total_head_m = head_loss_m + static_head_m

        # Flow rate
        flow_m3_hr = slurry_req['recommended_flow_m3_hr']
        flow_m3_s = flow_m3_hr / 3600

        # Pump power (hydraulic)
        # P = ρ × g × Q × H
        slurry_density = 1250  # kg/m³ (typical return slurry)
        hydraulic_power_W = slurry_density * g * flow_m3_s * total_head_m
        hydraulic_power_kW = hydraulic_power_W / 1000

        # Motor power (assuming 70% pump efficiency)
        pump_efficiency = 0.70
        motor_power_kW = hydraulic_power_kW / pump_efficiency

        return {
            'pipe_length_m': pipe_length_m,
            'friction_head_loss_m': head_loss_m,
            'static_head_m': static_head_m,
            'total_head_m': total_head_m,
            'flow_m3_hr': flow_m3_hr,
            'slurry_density_kg_m3': slurry_density,
            'hydraulic_power_kW': hydraulic_power_kW,
            'pump_efficiency': pump_efficiency,
            'motor_power_kW': motor_power_kW,
        }

    def full_calculation(self,
                        machine: MachineParams,
                        slurry_pipe: SlurryPipeParams,
                        soil_type: str = 'sand',
                        pipe_length_m: float = 200) -> Dict:
        """
        Perform complete slurry system calculation

        Args:
            machine: Machine parameters
            slurry_pipe: Slurry pipe parameters
            soil_type: Type of soil (sand, clay, gravel, mixed, rock)
            pipe_length_m: Total slurry pipe length

        Returns:
            Complete calculation results
        """
        # Get soil parameters
        soil = SOIL_PRESETS.get(soil_type.lower(), SOIL_PRESETS['mixed'])

        # Step 1: Excavation calculations
        excavation = self.calculate_excavation(machine)

        # Step 2: Slurry requirements
        slurry_req = self.calculate_slurry_requirements(excavation, soil)

        # Step 3: Pipe verification
        pipe_check = self.verify_pipe_sizing(slurry_req, slurry_pipe, soil)

        # Step 4: Pump requirements
        pump_req = self.calculate_pump_requirements(slurry_req, pipe_check, pipe_length_m)

        return {
            'excavation': excavation,
            'slurry_requirements': slurry_req,
            'pipe_verification': pipe_check,
            'pump_requirements': pump_req,
        }


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def main():
    """Demonstrate the complete slurry flow calculator"""

    print_header("COMPLETE SLURRY FLOW CALCULATOR")
    print("  For Microtunneling / Pipejacking Operations")
    print("  Based on Excavation Rate Methodology")

    # Initialize calculator
    calc = SlurryFlowCalculator()

    # Example parameters (typical AVN1200 machine)
    machine = MachineParams(
        cutter_diameter_mm=1500,      # 1500mm cutter head
        pipe_od_mm=1400,              # 1400mm OD pipe (DN1200)
        advance_rate_mm_min=30        # 30 mm/min advance rate
    )

    slurry_pipe = SlurryPipeParams(
        feed_pipe_diameter_mm=150,    # 150mm feed pipe
        return_pipe_diameter_mm=150   # 150mm return pipe
    )

    # Calculate for different soil types
    soil_types = ['sand', 'clay', 'gravel', 'mixed']

    print_header("INPUT PARAMETERS")
    print(f"""
  MACHINE PARAMETERS:
  ────────────────────────────────────────
  Cutter head diameter:    {machine.cutter_diameter_mm} mm
  Pipe outer diameter:     {machine.pipe_od_mm} mm
  Advance rate:            {machine.advance_rate_mm_min} mm/min

  SLURRY PIPE PARAMETERS:
  ────────────────────────────────────────
  Feed pipe diameter:      {slurry_pipe.feed_pipe_diameter_mm} mm
  Return pipe diameter:    {slurry_pipe.return_pipe_diameter_mm} mm
""")

    # Detailed calculation for sand
    print_header("DETAILED CALCULATION (SAND)")

    results = calc.full_calculation(machine, slurry_pipe, 'sand', 200)

    exc = results['excavation']
    print_subheader("1. EXCAVATION CALCULATIONS")
    print(f"""
  Cutter head area:        {exc['cutter_area_m2']:.4f} m²
  Pipe area:               {exc['pipe_area_m2']:.4f} m²
  Overcut (per side):      {exc['overcut_mm']:.1f} mm
  Overcut area:            {exc['overcut_area_m2']:.4f} m²

  Advance rate:            {exc['advance_rate_mm_min']:.1f} mm/min
                           {exc['advance_rate_m_hr']:.2f} m/hr

  EXCAVATION VOLUME RATE:
  ────────────────────────────────────────
  In-situ volume:          {exc['excavation_m3_hr']:.3f} m³/hr
  Overcut volume:          {exc['overcut_m3_hr']:.4f} m³/hr
""")

    slurry = results['slurry_requirements']
    print_subheader("2. SLURRY FLOW REQUIREMENTS")
    print(f"""
  Soil type:               {slurry['soil_type']}
  Bulking factor:          {slurry['bulking_factor']}
  Max concentration (Cv):  {slurry['max_concentration_Cv']*100:.0f}%
  Min transport velocity:  {slurry['min_velocity_mps']} m/s

  VOLUME CALCULATIONS:
  ────────────────────────────────────────
  In-situ volume:          {slurry['insitu_volume_m3_hr']:.3f} m³/hr
  Bulked volume:           {slurry['bulked_volume_m3_hr']:.3f} m³/hr

  REQUIRED SLURRY FLOW:
  ────────────────────────────────────────
  Minimum required:        {slurry['required_flow_m3_hr']:.2f} m³/hr
                           {slurry['required_flow_L_min']:.1f} L/min

  With safety factor ({slurry['safety_factor']}):
  Recommended flow:        {slurry['recommended_flow_m3_hr']:.2f} m³/hr
                           {slurry['recommended_flow_L_min']:.1f} L/min
""")

    pipes = results['pipe_verification']
    print_subheader("3. PIPE VERIFICATION")

    for pipe_type in ['feed', 'return']:
        p = pipes[pipe_type]
        status_symbol = "✓" if p['velocity_ok'] else "✗"
        print(f"""
  {pipe_type.upper()} PIPE:
  ────────────────────────────────────────
  Diameter:                {p['diameter_mm']} mm
  Cross-sectional area:    {p['area_m2']:.6f} m²

  Calculated velocity:     {p['velocity_mps']:.2f} m/s
  Required velocity:       {p['min_velocity_required']:.1f} - {p['max_velocity_allowed']:.1f} m/s
  Status:                  {status_symbol} {p['velocity_status']}

  Optimal pipe diameter:   {p['optimal_diameter_mm']:.0f} mm
""")

    pump = results['pump_requirements']
    print_subheader("4. PUMP REQUIREMENTS")
    print(f"""
  Pipe length:             {pump['pipe_length_m']} m

  HEAD CALCULATIONS:
  ────────────────────────────────────────
  Friction head loss:      {pump['friction_head_loss_m']:.2f} m
  Static head:             {pump['static_head_m']:.1f} m
  Total head required:     {pump['total_head_m']:.2f} m

  POWER CALCULATIONS:
  ────────────────────────────────────────
  Flow rate:               {pump['flow_m3_hr']:.2f} m³/hr
  Slurry density:          {pump['slurry_density_kg_m3']} kg/m³
  Hydraulic power:         {pump['hydraulic_power_kW']:.2f} kW
  Pump efficiency:         {pump['pump_efficiency']*100:.0f}%
  Motor power required:    {pump['motor_power_kW']:.2f} kW
""")

    # Summary table for all soil types
    print_header("COMPARISON: ALL SOIL TYPES")
    print(f"""
  Machine: {machine.cutter_diameter_mm}mm cutter, {machine.advance_rate_mm_min} mm/min advance
  Slurry pipe: {slurry_pipe.return_pipe_diameter_mm}mm diameter

  {'Soil Type':<15} {'Bulking':<10} {'Max Cv':<10} {'Min Vel':<10} {'Req Flow':<15} {'Velocity':<12} {'Status':<10}
  {'─'*82}""")

    for soil_type in soil_types:
        r = calc.full_calculation(machine, slurry_pipe, soil_type, 200)
        s = r['slurry_requirements']
        p = r['pipe_verification']['return']

        status = "✓ OK" if p['velocity_ok'] else f"✗ {p['velocity_status']}"

        print(f"  {s['soil_type']:<15} {s['bulking_factor']:<10.2f} {s['max_concentration_Cv']*100:<10.0f}% {s['min_velocity_mps']:<10.1f} {s['recommended_flow_m3_hr']:<10.1f} m³/hr  {p['velocity_mps']:<8.2f} m/s  {status}")

    # Formulas reference
    print_header("FORMULAS USED")
    print("""
  1. EXCAVATION VOLUME:
     ─────────────────────────────────────────────────────────────
     Cutter Area (m²) = π × (D_cutter / 2000)²

     Excavation Rate (m³/hr) = Cutter Area × Advance Rate (m/hr)

  2. SLURRY FLOW REQUIREMENT:
     ─────────────────────────────────────────────────────────────
     Bulked Volume = In-situ Volume × Bulking Factor

     Required Flow = Bulked Volume / Max Concentration (Cv)

     Recommended Flow = Required Flow × Safety Factor (1.3)

  3. PIPE VELOCITY:
     ─────────────────────────────────────────────────────────────
     Pipe Area (m²) = π × (D_pipe / 2000)²

     Velocity (m/s) = Flow Rate (m³/s) / Pipe Area (m²)

     Required: Min Velocity ≤ Actual Velocity ≤ 5.0 m/s

  4. PUMP POWER:
     ─────────────────────────────────────────────────────────────
     Head Loss (m) = f × (L/D) × (v² / 2g)

     Hydraulic Power (kW) = ρ × g × Q × H / 1000

     Motor Power (kW) = Hydraulic Power / Efficiency

  5. SOIL PARAMETERS:
     ─────────────────────────────────────────────────────────────
     Soil Type     Bulking Factor    Max Cv    Min Velocity
     Sand          1.15              25%       3.0 m/s
     Clay          1.25              20%       2.5 m/s
     Gravel        1.10              15%       3.5 m/s
     Mixed         1.20              20%       3.0 m/s
     Rock          1.40              15%       4.0 m/s
""")

    print_header("CALCULATION COMPLETE")
    print("\n  This calculator accounts for actual excavation volume!")
    print("  Use recommended flow rates for safe operation.\n")

    # =========================================================================
    # NEW FEATURES DEMONSTRATION
    # =========================================================================

    print_header("NEW FEATURES: ADVANCED SLURRY CALCULATIONS")

    # 1. Settling Velocity Calculation
    print_subheader("1. PARTICLE SETTLING VELOCITY")

    particles = [
        ParticleParams(d50_mm=0.1, particle_sg=2.65, shape_factor=0.8),   # Fine sand
        ParticleParams(d50_mm=0.5, particle_sg=2.65, shape_factor=0.7),   # Medium sand
        ParticleParams(d50_mm=2.0, particle_sg=2.65, shape_factor=0.6),   # Coarse sand
        ParticleParams(d50_mm=5.0, particle_sg=2.65, shape_factor=0.5),   # Gravel
    ]

    print(f"""
  Settling velocity determines how fast particles fall through fluid.
  Critical for understanding transport behavior in slurry pipelines.

  {'Particle Size':<15} {'SG':<8} {'Settling Vel':<15} {'Flow Regime':<25}
  {'─'*70}""")

    for p in particles:
        settling = calc.calculate_settling_velocity(p)
        print(f"  {settling['particle_diameter_mm']:<15.1f} {settling['particle_sg']:<8.2f} "
              f"{settling['settling_velocity_mm_s']:<12.2f} mm/s  {settling['flow_regime']:<25}")

    # 2. Critical Velocity Calculation
    print_subheader("2. CRITICAL DEPOSITION VELOCITY")

    particle = ParticleParams(d50_mm=0.5, particle_sg=2.65, shape_factor=0.7)
    critical = calc.calculate_critical_velocity(
        pipe_diameter_mm=150,
        particle=particle,
        Cv=0.20
    )

    print(f"""
  Critical velocity is the minimum flow speed to prevent particle settling.
  Below this velocity, solids will deposit and block the pipe.

  DURAND-CONDOLIOS METHOD:
  ────────────────────────────────────────
  Pipe diameter:           {critical['pipe_diameter_mm']} mm
  Particle d50:            {critical['particle_d50_mm']} mm
  Volume concentration:    {critical['volume_concentration_Cv']*100:.0f}%
  Durand factor (F_L):     {critical['durand_factor_FL']:.3f}

  RESULTS:
  ────────────────────────────────────────
  Critical velocity (Durand):  {critical['critical_velocity_durand_mps']:.2f} m/s
  Critical velocity (Wasp):    {critical['critical_velocity_wasp_mps']:.2f} m/s
  Recommended minimum:         {critical['recommended_minimum_mps']:.2f} m/s (with 20% safety)
""")

    # 3. Mass Balance Calculation
    print_subheader("3. EXCAVATION MASS BALANCE")

    circuit = SlurryCircuitParams(
        Q_in_m3_hr=200,      # Feed flow 200 m³/hr
        Q_out_m3_hr=205,     # Return flow 205 m³/hr
        rho_in=1080,         # Feed density 1.08 g/cm³
        rho_out=1250,        # Return density 1.25 g/cm³
        rho_solids=2650,     # Silica
        rho_water=1000       # Water
    )

    mass_balance = calc.calculate_excavation_mass_balance(circuit, time_hours=1.0)

    print(f"""
  Based on Duhme et al. (2016): Mass_in + Mass_excavated = Mass_out

  SLURRY CIRCUIT DATA:
  ────────────────────────────────────────
  Feed flow rate:          {mass_balance['flow_rate_in_m3_hr']:.1f} m³/hr
  Return flow rate:        {mass_balance['flow_rate_out_m3_hr']:.1f} m³/hr
  Feed density:            {mass_balance['density_in_kg_m3']:.0f} kg/m³
  Return density:          {mass_balance['density_out_kg_m3']:.0f} kg/m³

  CALCULATED CONCENTRATIONS:
  ────────────────────────────────────────
  Feed Cv:                 {mass_balance['concentration_Cv_in']*100:.1f}%
  Return Cv:               {mass_balance['concentration_Cv_out']*100:.1f}%

  EXCAVATION RESULTS (per hour):
  ────────────────────────────────────────
  Excavated volume:        {mass_balance['excavated_volume_m3_hr']:.3f} m³/hr
  Excavated dry mass:      {mass_balance['excavated_dry_mass_kg_hr']:.1f} kg/hr
                           {mass_balance['total_excavated_mass_tonnes']:.3f} tonnes/hr
""")

    # 4. Over-excavation Check
    print_subheader("4. OVER-EXCAVATION MONITORING")

    # Theoretical volume from excavation rate
    theoretical_vol = results['excavation']['excavation_m3_hr']  # From earlier calculation
    measured_vol = mass_balance['excavated_volume_m3_hr']

    # Scale to same basis (use theoretical as reference)
    overexc_check = calc.check_overexcavation(
        measured_volume_m3=measured_vol,
        theoretical_volume_m3=theoretical_vol,
        tolerance_percent=10.0
    )

    print(f"""
  Comparing measured excavation (from slurry circuit) vs theoretical.

  COMPARISON:
  ────────────────────────────────────────
  Theoretical volume:      {overexc_check['theoretical_volume_m3']:.3f} m³/hr
  Measured volume:         {overexc_check['measured_volume_m3']:.3f} m³/hr
  Difference:              {overexc_check['difference_m3']:.3f} m³/hr ({overexc_check['difference_percent']:.1f}%)
  Tolerance:               ±{overexc_check['tolerance_percent']:.0f}%

  STATUS: {overexc_check['status']}
  {overexc_check['recommendation']}
""")

    # 5. Bentonite Injection
    print_subheader("5. BENTONITE INJECTION REQUIREMENTS")

    print(f"""
  Bentonite injection rates by ground type (based on Praetorius & Schößer 2017):

  {'Ground Type':<15} {'Rate (L/m/mD)':<18} {'For 1400mm Pipe':<18} {'Description':<25}
  {'─'*80}""")

    for ground_type, data in BENTONITE_INJECTION.items():
        rate = data['rate_L_per_m_per_mD']
        for_1400 = rate * 1.4  # 1400mm diameter
        print(f"  {ground_type.capitalize():<15} {rate:<18.0f} {for_1400:<15.1f} L/m  {data['description']:<25}")

    # Detailed calculation for sand
    bento = calc.calculate_bentonite_injection(
        pipe_od_mm=1400,
        tunnel_length_m=500,
        ground_type='sand',
        advance_rate_m_hr=1.8
    )

    print(f"""

  EXAMPLE: DN1200 pipe (OD 1400mm), 500m tunnel in sand
  ────────────────────────────────────────
  Ground type:             {bento['ground_type']} ({bento['ground_description']})
  Injection rate:          {bento['injection_rate_L_per_m']:.1f} L/m of tunnel
  Total mix required:      {bento['total_bentonite_mix_L']:.0f} L ({bento['total_bentonite_mix_m3']:.1f} m³)
  Bentonite powder:        {bento['bentonite_powder_kg']:.0f} kg
  Water required:          {bento['water_required_L']:.0f} L

  At {bento['advance_rate_m_hr']:.1f} m/hr advance rate:
  Pumping rate:            {bento['pumping_rate_L_hr']:.1f} L/hr ({bento['pumping_rate_L_min']:.2f} L/min)
""")

    # 6. Slurry Head Loss
    print_subheader("6. SLURRY PIPELINE HEAD LOSS")

    particle_sand = ParticleParams(d50_mm=0.5, particle_sg=2.65)
    head_loss = calc.calculate_slurry_head_loss(
        pipe_diameter_mm=150,
        pipe_length_m=200,
        flow_rate_m3_hr=250,
        slurry_density_kg_m3=1250,
        particle=particle_sand,
        Cv=0.15
    )

    print(f"""
  Slurry head loss is greater than water due to particle transport.
  Uses Durand-Condolios correction method.

  PIPELINE PARAMETERS:
  ────────────────────────────────────────
  Pipe diameter:           {head_loss['pipe_diameter_mm']} mm
  Pipe length:             {head_loss['pipe_length_m']} m
  Flow rate:               {head_loss['flow_rate_m3_hr']:.1f} m³/hr
  Velocity:                {head_loss['velocity_mps']:.2f} m/s
  Reynolds number:         {head_loss['reynolds_number']:.0f}

  SLURRY PROPERTIES:
  ────────────────────────────────────────
  Slurry density:          {head_loss['slurry_density_kg_m3']} kg/m³
  Volume concentration:    {head_loss['volume_concentration_Cv']*100:.0f}%
  Particle d50:            {particle_sand.d50_mm} mm

  HEAD LOSS RESULTS:
  ────────────────────────────────────────
  Friction factor (water): {head_loss['friction_factor_water']:.4f}
  Head loss (water only):  {head_loss['head_loss_water_m']:.2f} m
  Durand K factor:         {head_loss['durand_K_factor']}
  Slurry correction (φ):   {head_loss['slurry_correction_phi']:.2f}
  Head loss (slurry):      {head_loss['head_loss_slurry_m']:.2f} m
  Head loss increase:      +{head_loss['head_loss_increase_percent']:.0f}%
  Pressure loss:           {head_loss['pressure_loss_bar']:.2f} bar
""")

    # Summary of new formulas
    print_header("NEW FORMULAS REFERENCE")
    print("""
  1. SETTLING VELOCITY (Stokes Law - laminar):
     ─────────────────────────────────────────────────────────────
     v_s = g × d² × (ρ_s - ρ_f) / (18 × μ)

     Where: g = gravity, d = particle diameter, μ = viscosity

  2. CRITICAL VELOCITY (Durand-Condolios):
     ─────────────────────────────────────────────────────────────
     V_c = F_L × √(2gD(S-1))

     Where: F_L = Durand factor (0.8-2.0), D = pipe diameter
            S = particle specific gravity

  3. MASS BALANCE (Duhme et al.):
     ─────────────────────────────────────────────────────────────
     Mass_in + Mass_excavated = Mass_out
     Cv = (ρ_slurry - ρ_water) / (ρ_solids - ρ_water)

  4. BENTONITE INJECTION:
     ─────────────────────────────────────────────────────────────
     Rate (L/m) = Base Rate × Pipe Diameter (m)
     Base rates: Clay=15, Silt=25, Sand=40, Gravel=60 L/m/mD

  5. SLURRY HEAD LOSS (Durand-Condolios correction):
     ─────────────────────────────────────────────────────────────
     φ = 1 + K × Cv × [(gD(S-1))/V²]^1.5
     h_slurry = h_water × φ

     Where: K = 81 (fine), 121 (medium), 150 (coarse)
""")

    print_header("ALL CALCULATIONS COMPLETE")
    print("\n  Enhanced slurry calculator with advanced features!")
    print("  Based on industry research and engineering standards.\n")


if __name__ == "__main__":
    main()
