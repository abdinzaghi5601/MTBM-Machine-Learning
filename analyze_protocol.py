#!/usr/bin/env python3
"""
Multi-Protocol MTBM Analysis Tool
==================================

Universal analysis tool that works with any AVN protocol:
- AVN 800
- AVN 1200
- AVN 2400
- AVN 3000

Usage:
    python analyze_protocol.py --protocol AVN2400 --data ../data/raw/my_data.csv
    python analyze_protocol.py --protocol AVN800 --generate-sample

Author: MTBM ML Framework
Date: November 2024
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import our configurations and plotter
from protocol_configs import get_protocol_config, SUPPORTED_PROTOCOLS
from mtbm_comprehensive_plotting import MTBMComprehensivePlotter


class ProtocolAnalyzer:
    """
    Multi-protocol analyzer that adapts to different AVN protocols
    """

    def __init__(self, protocol_name: str, base_dir: Path = None):
        """
        Initialize analyzer for specific protocol

        Args:
            protocol_name: One of AVN800, AVN1200, AVN2400, AVN3000
            base_dir: Base directory for outputs
        """
        self.protocol_name = protocol_name
        self.config = get_protocol_config(protocol_name)

        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = base_dir

        # Create protocol-specific output directories
        self.protocol_dir = base_dir / 'outputs' / protocol_name.replace(' ', '_')
        self.plots_dir = self.protocol_dir / 'plots'
        self.data_dir = self.protocol_dir / 'data'
        self.reports_dir = self.protocol_dir / 'reports'

        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialized {self.config.protocol_name} Analyzer")
        print(f"Outputs will be saved to: {self.protocol_dir}")

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            data_path: Path to CSV file

        Returns:
            Loaded DataFrame
        """
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Convert timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)[:10]}...")

        return df

    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample data compatible with this protocol

        Args:
            n_samples: Number of samples to generate

        Returns:
            Generated DataFrame
        """
        print(f"\nGenerating {n_samples} sample records for {self.config.protocol_name}")

        np.random.seed(42)

        # Use the base plotter to generate data
        plotter = MTBMComprehensivePlotter()
        df = plotter.generate_synthetic_mtbm_data(n_samples=n_samples)

        # Filter to only parameters supported by this protocol
        available_params = self.config.get_parameter_names()

        # Keep timestamp and date/time columns
        keep_cols = ['timestamp', 'date', 'time'] + [
            col for col in df.columns if col in available_params
        ]

        df = df[keep_cols]

        print(f"Generated data with {len(df.columns)} parameters")
        return df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate data against protocol configuration

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        print(f"\nValidating data against {self.config.protocol_name} specifications...")

        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'parameter_checks': {}
        }

        for param_name, param_config in self.config.parameters.items():
            if param_name not in df.columns:
                validation_results['warnings'].append(
                    f"Parameter '{param_name}' not found in data (optional for {self.config.protocol_name})"
                )
                continue

            # Check value ranges
            param_data = df[param_name]
            min_val = param_data.min()
            max_val = param_data.max()

            param_check = {
                'min': min_val,
                'max': max_val,
                'within_range': True,
                'within_normal': True
            }

            # Check absolute limits
            if min_val < param_config.min_value or max_val > param_config.max_value:
                param_check['within_range'] = False
                validation_results['errors'].append(
                    f"{param_name}: Values outside allowed range "
                    f"[{param_config.min_value}, {param_config.max_value}]. "
                    f"Found: [{min_val:.2f}, {max_val:.2f}]"
                )

            # Check normal operating range
            if min_val < param_config.normal_min or max_val > param_config.normal_max:
                param_check['within_normal'] = False
                validation_results['warnings'].append(
                    f"{param_name}: Some values outside normal range "
                    f"[{param_config.normal_min}, {param_config.normal_max}]. "
                    f"Found: [{min_val:.2f}, {max_val:.2f}]"
                )

            validation_results['parameter_checks'][param_name] = param_check

        if validation_results['errors']:
            validation_results['valid'] = False

        # Print summary
        print(f"Validation complete:")
        print(f"  Errors: {len(validation_results['errors'])}")
        print(f"  Warnings: {len(validation_results['warnings'])}")

        if not validation_results['valid']:
            print("\nERRORS:")
            for error in validation_results['errors']:
                print(f"  ❌ {error}")

        if validation_results['warnings']:
            print("\nWARNINGS:")
            for warning in validation_results['warnings'][:5]:  # Show first 5
                print(f"  ⚠️  {warning}")
            if len(validation_results['warnings']) > 5:
                print(f"  ... and {len(validation_results['warnings']) - 5} more warnings")

        return validation_results

    def analyze(self, df: pd.DataFrame, save_plots: bool = True):
        """
        Run complete analysis on data

        Args:
            df: DataFrame with MTBM data
            save_plots: Whether to save plot files
        """
        print(f"\n{'='*70}")
        print(f"Running {self.config.protocol_name} Analysis")
        print(f"{'='*70}")

        # Create a modified plotter that uses our directories
        plotter = MTBMComprehensivePlotter(base_dir=self.base_dir)

        # Override directories to use protocol-specific ones
        plotter.plots_dir = self.plots_dir
        plotter.processed_data_dir = self.data_dir
        plotter.reports_dir = self.reports_dir

        # Update deviation thresholds based on protocol
        plotter.deviation_thresholds = self.config.deviation_thresholds

        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.data_dir / f'{self.config.protocol_name.replace(" ", "_")}_data_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved processed data: {csv_path}")

        # Generate all visualizations
        print(f"\n📊 Generating visualizations...")

        # 1. Time Series
        print("  Creating time series overview...")
        plotter.plot_time_series_overview(df, save_plots=save_plots)

        # 2. Deviation Analysis (if deviation parameters exist)
        if 'hor_deviation_machine_mm' in df.columns and 'vert_deviation_machine_mm' in df.columns:
            print("  Creating deviation analysis...")

            # Calculate total deviation if not present
            if 'total_deviation_mm' not in df.columns:
                df['total_deviation_mm'] = np.sqrt(
                    df['hor_deviation_machine_mm']**2 +
                    df['vert_deviation_machine_mm']**2
                )

            plotter.plot_deviation_analysis(df, save_plots=save_plots)
        else:
            print("  ⚠️  Skipping deviation analysis (parameters not available)")

        # 3. Performance Dashboard (if key parameters exist)
        required_for_performance = ['advance_speed_mm_min', 'working_pressure_bar']
        if all(param in df.columns for param in required_for_performance):
            print("  Creating performance dashboard...")

            # Calculate efficiency metrics if not present
            if 'drilling_efficiency' not in df.columns and 'working_pressure_bar' in df.columns:
                df['drilling_efficiency'] = df['advance_speed_mm_min'] / df['working_pressure_bar']

            if 'power_efficiency' not in df.columns and 'revolution_rpm' in df.columns:
                df['power_efficiency'] = df['advance_speed_mm_min'] / df['revolution_rpm']

            plotter.plot_performance_dashboard(df, save_plots=save_plots)
        else:
            print("  ⚠️  Skipping performance dashboard (parameters not available)")

        # 4. Correlation Matrix
        print("  Creating correlation matrix...")
        plotter.plot_correlation_matrix(df, save_plots=save_plots)

        # 5. Generate text report
        print("\n📄 Generating analysis report...")
        self.generate_protocol_report(df, plotter)

        print(f"\n{'='*70}")
        print(f"✅ Analysis Complete!")
        print(f"{'='*70}")
        print(f"\nOutputs saved to:")
        print(f"  📊 Plots: {self.plots_dir}")
        print(f"  📁 Data: {self.data_dir}")
        print(f"  📄 Reports: {self.reports_dir}")

    def generate_protocol_report(self, df: pd.DataFrame, plotter):
        """
        Generate protocol-specific analysis report

        Args:
            df: Analyzed DataFrame
            plotter: Plotter instance
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = self.reports_dir / f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"{self.config.protocol_name} ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Protocol: {self.config.protocol_name}\n")
            f.write(f"Records: {len(df):,}\n\n")

            # Data Overview
            f.write("1. DATA OVERVIEW\n")
            f.write("-" * 40 + "\n")
            if 'timestamp' in df.columns:
                f.write(f"Time Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            if 'tunnel_length_m' in df.columns:
                f.write(f"Tunnel Length: {df['tunnel_length_m'].max():.1f} meters\n")
            f.write(f"Parameters: {len(df.columns)}\n\n")

            # Protocol-specific quality thresholds
            f.write("2. QUALITY THRESHOLDS (Protocol Specific)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Deviation - Excellent: ≤ {self.config.deviation_thresholds['excellent']}mm\n")
            f.write(f"Deviation - Good: ≤ {self.config.deviation_thresholds['good']}mm\n")
            f.write(f"Deviation - Poor: > {self.config.deviation_thresholds['poor']}mm\n\n")

            # Parameter statistics
            f.write("3. PARAMETER STATISTICS\n")
            f.write("-" * 40 + "\n")
            for param_name, param_config in self.config.parameters.items():
                if param_name in df.columns:
                    param_data = df[param_name]
                    f.write(f"\n{param_config.display_name} ({param_config.unit}):\n")
                    f.write(f"  Min: {param_data.min():.2f}\n")
                    f.write(f"  Max: {param_data.max():.2f}\n")
                    f.write(f"  Mean: {param_data.mean():.2f}\n")
                    f.write(f"  Normal Range: {param_config.normal_min:.1f} - {param_config.normal_max:.1f}\n")

                    # Check if within normal range
                    in_range = (param_data >= param_config.normal_min) & (param_data <= param_config.normal_max)
                    pct_in_range = (in_range.sum() / len(param_data)) * 100
                    f.write(f"  Within Normal: {pct_in_range:.1f}%\n")

            # Use base plotter report for additional metrics
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED OPERATIONAL ANALYSIS\n")
            f.write("="*80 + "\n\n")

        # Append the comprehensive report
        print(f"✅ Saved analysis report: {report_path}")

        # Also print to console
        plotter.generate_comprehensive_report(df)


def main():
    """Main entry point for multi-protocol analysis"""

    parser = argparse.ArgumentParser(
        description='MTBM Multi-Protocol Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing data with AVN 2400 protocol:
  python analyze_protocol.py --protocol AVN2400 --data my_data.csv

  # Generate and analyze sample data for AVN 800:
  python analyze_protocol.py --protocol AVN800 --generate-sample

  # Analyze with specific number of samples:
  python analyze_protocol.py --protocol AVN3000 --generate-sample --samples 5000
        """
    )

    parser.add_argument(
        '--protocol',
        type=str,
        required=True,
        choices=SUPPORTED_PROTOCOLS,
        help='AVN protocol to use'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to CSV data file'
    )

    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate sample synthetic data'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples to generate (with --generate-sample)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation (data analysis only)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data, do not run analysis'
    )

    args = parser.parse_args()

    # Create analyzer for specified protocol
    analyzer = ProtocolAnalyzer(args.protocol)

    # Load or generate data
    if args.generate_sample:
        df = analyzer.generate_sample_data(n_samples=args.samples)
    elif args.data:
        df = analyzer.load_data(args.data)
    else:
        print("Error: Must specify either --data or --generate-sample")
        sys.exit(1)

    # Validate data
    validation = analyzer.validate_data(df)

    if not validation['valid'] and not args.validate_only:
        print("\n⚠️  Data validation failed. Proceeding anyway...")
        response = input("Continue with analysis? (y/n): ")
        if response.lower() != 'y':
            print("Analysis cancelled.")
            sys.exit(1)

    if args.validate_only:
        print("\n✅ Validation complete. Exiting (--validate-only specified).")
        sys.exit(0)

    # Run analysis
    analyzer.analyze(df, save_plots=not args.no_plots)


if __name__ == "__main__":
    main()
