#!/usr/bin/env python3
"""
MTBM Protocol Configurations
=============================

Configuration definitions for different AVN protocols:
- AVN 800
- AVN 1200
- AVN 2400
- AVN 3000

Each protocol may have different:
- Available parameters
- Data ranges
- Sampling rates
- Quality thresholds

Author: MTBM ML Framework
Date: November 2024
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ParameterConfig:
    """Configuration for a single parameter"""
    name: str
    display_name: str
    unit: str
    min_value: float
    max_value: float
    normal_min: float
    normal_max: float
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None


class ProtocolConfig:
    """Base configuration for MTBM protocols"""

    def __init__(self, protocol_name: str):
        self.protocol_name = protocol_name
        self.parameters = {}
        self.deviation_thresholds = {
            'excellent': 25,  # mm
            'good': 50,       # mm
            'poor': 75        # mm
        }

    def add_parameter(self, param: ParameterConfig):
        """Add a parameter to this protocol configuration"""
        self.parameters[param.name] = param

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names"""
        return list(self.parameters.keys())

    def get_display_name(self, param_name: str) -> str:
        """Get display name for a parameter"""
        return self.parameters[param_name].display_name if param_name in self.parameters else param_name


class AVN800Config(ProtocolConfig):
    """AVN 800 Protocol Configuration"""

    def __init__(self):
        super().__init__("AVN 800")

        # Survey Position Parameters
        self.add_parameter(ParameterConfig(
            name='tunnel_length_m',
            display_name='Tunnel Length',
            unit='m',
            min_value=0,
            max_value=10000,
            normal_min=0,
            normal_max=10000
        ))

        self.add_parameter(ParameterConfig(
            name='hor_deviation_machine_mm',
            display_name='Horizontal Deviation (Machine)',
            unit='mm',
            min_value=-200,
            max_value=200,
            normal_min=-50,
            normal_max=50,
            critical_min=-100,
            critical_max=100
        ))

        self.add_parameter(ParameterConfig(
            name='vert_deviation_machine_mm',
            display_name='Vertical Deviation (Machine)',
            unit='mm',
            min_value=-200,
            max_value=200,
            normal_min=-50,
            normal_max=50,
            critical_min=-100,
            critical_max=100
        ))

        # Steering Control
        for i in range(1, 5):
            self.add_parameter(ParameterConfig(
                name=f'cylinder_{i:02d}_stroke_mm',
                display_name=f'Cylinder {i} Stroke',
                unit='mm',
                min_value=0,
                max_value=100,
                normal_min=20,
                normal_max=80
            ))

        # Operational Parameters
        self.add_parameter(ParameterConfig(
            name='advance_speed_mm_min',
            display_name='Advance Speed',
            unit='mm/min',
            min_value=0,
            max_value=60,
            normal_min=15,
            normal_max=45,
            critical_min=10,
            critical_max=55
        ))

        self.add_parameter(ParameterConfig(
            name='working_pressure_bar',
            display_name='Working Pressure',
            unit='bar',
            min_value=0,
            max_value=250,
            normal_min=120,
            normal_max=200,
            critical_min=100,
            critical_max=220
        ))

        self.add_parameter(ParameterConfig(
            name='revolution_rpm',
            display_name='Revolution Speed',
            unit='RPM',
            min_value=0,
            max_value=20,
            normal_min=6,
            normal_max=12,
            critical_min=4,
            critical_max=15
        ))

        self.add_parameter(ParameterConfig(
            name='earth_pressure_01_bar',
            display_name='Earth Pressure',
            unit='bar',
            min_value=-10,
            max_value=30,
            normal_min=8,
            normal_max=26,
            critical_min=5,
            critical_max=30
        ))


class AVN1200Config(ProtocolConfig):
    """AVN 1200 Protocol Configuration"""

    def __init__(self):
        super().__init__("AVN 1200")

        # Inherits most from AVN 800 but with some differences
        # Start with AVN 800 parameters
        avn800 = AVN800Config()
        for param in avn800.parameters.values():
            self.add_parameter(param)

        # Add/modify specific AVN 1200 parameters
        self.add_parameter(ParameterConfig(
            name='yaw_mm_per_m',
            display_name='Yaw',
            unit='mm/m',
            min_value=-10,
            max_value=10,
            normal_min=-5,
            normal_max=5
        ))

        self.add_parameter(ParameterConfig(
            name='pitch_mm_per_m',
            display_name='Pitch',
            unit='mm/m',
            min_value=-10,
            max_value=10,
            normal_min=-5,
            normal_max=5
        ))

        # AVN 1200 specific: Enhanced survey capabilities
        self.add_parameter(ParameterConfig(
            name='temperature_els_mwd',
            display_name='ELS/MWD Temperature',
            unit='°C',
            min_value=-20,
            max_value=60,
            normal_min=15,
            normal_max=30,
            critical_min=10,
            critical_max=35
        ))


class AVN2400Config(ProtocolConfig):
    """AVN 2400 Protocol Configuration - Enhanced Measurement System"""

    def __init__(self):
        super().__init__("AVN 2400")

        # Start with AVN 1200 base
        avn1200 = AVN1200Config()
        for param in avn1200.parameters.values():
            self.add_parameter(param)

        # AVN 2400 additions: Drill head specific measurements
        self.add_parameter(ParameterConfig(
            name='hor_deviation_drill_head_mm',
            display_name='Horizontal Deviation (Drill Head)',
            unit='mm',
            min_value=-200,
            max_value=200,
            normal_min=-50,
            normal_max=50,
            critical_min=-100,
            critical_max=100
        ))

        self.add_parameter(ParameterConfig(
            name='vert_deviation_drill_head_mm',
            display_name='Vertical Deviation (Drill Head)',
            unit='mm',
            min_value=-200,
            max_value=200,
            normal_min=-50,
            normal_max=50,
            critical_min=-100,
            critical_max=100
        ))

        # Enhanced force measurements
        self.add_parameter(ParameterConfig(
            name='total_force_kn',
            display_name='Total Steering Force',
            unit='kN',
            min_value=0,
            max_value=1000,
            normal_min=400,
            normal_max=750,
            critical_min=300,
            critical_max=900
        ))

        self.add_parameter(ParameterConfig(
            name='interjack_force_kn',
            display_name='Interjack Force',
            unit='kN',
            min_value=0,
            max_value=2000,
            normal_min=800,
            normal_max=1500,
            critical_min=600,
            critical_max=1700
        ))

        # Reel angle measurement
        self.add_parameter(ParameterConfig(
            name='reel_degree',
            display_name='Reel Angle',
            unit='°',
            min_value=-180,
            max_value=180,
            normal_min=-180,
            normal_max=180
        ))


class AVN3000Config(ProtocolConfig):
    """AVN 3000 Protocol Configuration - Full Feature Set"""

    def __init__(self):
        super().__init__("AVN 3000")

        # Start with AVN 2400 base (most comprehensive)
        avn2400 = AVN2400Config()
        for param in avn2400.parameters.values():
            self.add_parameter(param)

        # AVN 3000 specific: Advanced monitoring
        self.add_parameter(ParameterConfig(
            name='survey_mode',
            display_name='Survey Mode',
            unit='',
            min_value=0,
            max_value=2,
            normal_min=0,
            normal_max=2
        ))

        self.add_parameter(ParameterConfig(
            name='activated_interjack',
            display_name='Active Interjack',
            unit='',
            min_value=1,
            max_value=4,
            normal_min=1,
            normal_max=4
        ))

        # Enhanced thresholds for AVN 3000
        self.deviation_thresholds = {
            'excellent': 20,  # Tighter tolerance
            'good': 40,
            'poor': 60
        }

        # Tighter operating ranges (more precise control)
        self.parameters['advance_speed_mm_min'].normal_min = 20
        self.parameters['advance_speed_mm_min'].normal_max = 40

        self.parameters['earth_pressure_01_bar'].normal_min = 10
        self.parameters['earth_pressure_01_bar'].normal_max = 24


# Protocol factory
def get_protocol_config(protocol_name: str) -> ProtocolConfig:
    """
    Factory function to get the appropriate protocol configuration

    Args:
        protocol_name: One of 'AVN800', 'AVN1200', 'AVN2400', 'AVN3000'

    Returns:
        ProtocolConfig instance for the specified protocol

    Raises:
        ValueError: If protocol_name is not recognized
    """
    protocol_map = {
        'AVN800': AVN800Config,
        'AVN 800': AVN800Config,
        '800': AVN800Config,
        'AVN1200': AVN1200Config,
        'AVN 1200': AVN1200Config,
        '1200': AVN1200Config,
        'AVN2400': AVN2400Config,
        'AVN 2400': AVN2400Config,
        '2400': AVN2400Config,
        'AVN3000': AVN3000Config,
        'AVN 3000': AVN3000Config,
        '3000': AVN3000Config,
    }

    protocol_key = protocol_name.upper().replace(' ', '')

    if protocol_key not in protocol_map:
        available = list(set([k for k in protocol_map.keys() if 'AVN' in k]))
        raise ValueError(
            f"Unknown protocol: {protocol_name}. "
            f"Available protocols: {', '.join(available)}"
        )

    return protocol_map[protocol_key]()


# Quick reference
SUPPORTED_PROTOCOLS = ['AVN800', 'AVN1200', 'AVN2400', 'AVN3000']


if __name__ == "__main__":
    # Test the configurations
    print("MTBM Protocol Configurations")
    print("=" * 60)

    for protocol_name in SUPPORTED_PROTOCOLS:
        config = get_protocol_config(protocol_name)
        print(f"\n{config.protocol_name}")
        print(f"Parameters: {len(config.parameters)}")
        print(f"Deviation thresholds: {config.deviation_thresholds}")
        print(f"Sample parameters: {list(config.parameters.keys())[:5]}")
