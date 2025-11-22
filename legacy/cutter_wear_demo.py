#!/usr/bin/env python3
"""
MTBM Cutter Wear Prediction Demo
Simplified demonstration of the cutter wear prediction and geological correlation framework
using only built-in Python libraries.
"""

import math
import random
import csv
from datetime import datetime, timedelta

class CutterWearDemo:
    """
    Demonstration of MTBM cutter wear prediction and geological correlation system
    """
    
    def __init__(self):
        # Geological classification
        self.geological_types = {
            'soft_clay': {'wear_factor': 0.02, 'pressure': 80, 'force': 400, 'abrasivity': 2.0},
            'hard_clay': {'wear_factor': 0.05, 'pressure': 120, 'force': 600, 'abrasivity': 3.0},
            'sandy_clay': {'wear_factor': 0.08, 'pressure': 100, 'force': 550, 'abrasivity': 4.0},
            'loose_sand': {'wear_factor': 0.06, 'pressure': 60, 'force': 350, 'abrasivity': 3.5},
            'dense_sand': {'wear_factor': 0.12, 'pressure': 140, 'force': 700, 'abrasivity': 4.5},
            'gravel': {'wear_factor': 0.18, 'pressure': 160, 'force': 800, 'abrasivity': 5.5},
            'soft_rock': {'wear_factor': 0.15, 'pressure': 180, 'force': 900, 'abrasivity': 4.0},
            'medium_rock': {'wear_factor': 0.25, 'pressure': 220, 'force': 1100, 'abrasivity': 5.0},
            'hard_rock': {'wear_factor': 0.45, 'pressure': 280, 'force': 1400, 'abrasivity': 6.5},
            'mixed_ground': {'wear_factor': 0.20, 'pressure': 150, 'force': 750, 'abrasivity': 4.5},
            'weathered_rock': {'wear_factor': 0.10, 'pressure': 130, 'force': 650, 'abrasivity': 3.5}
        }
        
        # Cutter wear thresholds (mm)
        self.wear_thresholds = {
            'new': 0.0, 'light_wear': 2.0, 'moderate_wear': 5.0, 
            'heavy_wear': 8.0, 'replacement_needed': 12.0
        }
        
        # Cutter positions and their wear multipliers
        self.cutter_positions = {
            'face_center': 1.2, 'face_outer': 1.0, 'gauge': 1.5, 'back_reamer': 0.8
        }
        
        # Optimal parameters learned from simulation
        self.optimal_parameters = {}
        self.wear_history = []
        
    def generate_tunneling_data(self, tunnel_length=500, readings=1000):
        """Generate realistic tunneling data with geological conditions"""
        
        print(f"Generating {readings} readings for {tunnel_length}m tunnel...")
        
        data = []
        random.seed(42)  # For reproducible results
        
        # Define geological zones along tunnel
        geological_zones = [
            ('soft_clay', 0, 100), ('sandy_clay', 100, 200), 
            ('dense_sand', 200, 300), ('gravel', 300, 350),
            ('medium_rock', 350, 450), ('hard_rock', 450, 500)
        ]
        
        for i in range(readings):
            distance = (i / readings) * tunnel_length
            hours = i * 0.25  # 15 minutes per reading
            
            # Determine current geology
            current_geology = 'soft_clay'
            for geo_type, start, end in geological_zones:
                if start <= distance < end:
                    current_geology = geo_type
                    break
            
            geo_props = self.geological_types[current_geology]
            
            # Generate operational parameters with some variation
            advance_speed = 45 + random.uniform(-10, 10)
            revolution_rpm = 8.5 + random.uniform(-1.5, 1.5)
            working_pressure = 180 + random.uniform(-30, 30)
            earth_pressure = geo_props['pressure'] + random.uniform(-15, 15)
            total_force = geo_props['force'] + random.uniform(-100, 100)
            
            # Calculate wear for each cutter type
            for position, multiplier in self.cutter_positions.items():
                for cutter_id in range(1, 7):  # 6 cutters per position
                    
                    # Calculate wear rate
                    wear_rate = self.calculate_wear_rate(
                        current_geology, advance_speed, revolution_rpm, 
                        earth_pressure, total_force, hours
                    )
                    
                    # Apply position multiplier and individual variation
                    actual_wear_rate = wear_rate * multiplier * (0.8 + random.random() * 0.4)
                    cumulative_wear = actual_wear_rate * hours
                    
                    # Add some wear pattern variation
                    cumulative_wear += random.uniform(-0.5, 0.5)
                    cumulative_wear = max(0, cumulative_wear)
                    
                    data.append({
                        'reading_id': i + 1,
                        'cutter_id': f"{position}_{cutter_id}",
                        'tunnel_distance': round(distance, 2),
                        'operating_hours': round(hours, 2),
                        'geological_type': current_geology,
                        'cutter_position': position,
                        'advance_speed': round(advance_speed, 1),
                        'revolution_rpm': round(revolution_rpm, 1),
                        'working_pressure': round(working_pressure, 0),
                        'earth_pressure': round(earth_pressure, 0),
                        'total_force': round(total_force, 0),
                        'abrasivity_index': geo_props['abrasivity'],
                        'total_wear': round(cumulative_wear, 3),
                        'wear_rate': round(actual_wear_rate, 4),
                        'wear_condition': self.classify_wear_condition(cumulative_wear)
                    })
        
        self.wear_history = data
        print(f"Generated {len(data)} cutter wear records")
        return data
    
    def calculate_wear_rate(self, geology, advance_speed, rpm, earth_pressure, total_force, hours):
        """Calculate wear rate based on conditions"""
        
        base_rate = self.geological_types[geology]['wear_factor']
        
        # Operational factors
        speed_factor = (advance_speed / 45) ** 0.5
        rpm_factor = (rpm / 8.5) ** 0.3
        pressure_factor = (earth_pressure / 120) ** 0.7
        force_factor = (total_force / 800) ** 0.6
        
        # Time degradation
        time_factor = 1 + (hours / 1000) * 0.1
        
        wear_rate = base_rate * speed_factor * rpm_factor * pressure_factor * force_factor * time_factor
        return max(0.001, wear_rate)
    
    def classify_wear_condition(self, wear_amount):
        """Classify wear condition"""
        
        if wear_amount < self.wear_thresholds['light_wear']:
            return 'new'
        elif wear_amount < self.wear_thresholds['moderate_wear']:
            return 'light_wear'
        elif wear_amount < self.wear_thresholds['heavy_wear']:
            return 'moderate_wear'
        elif wear_amount < self.wear_thresholds['replacement_needed']:
            return 'heavy_wear'
        else:
            return 'replacement_needed'
    
    def analyze_geological_correlation(self):
        """Analyze correlation between geology and optimal parameters"""
        
        print("Analyzing geological correlation for parameter optimization...")
        
        # Group data by geological type
        geo_analysis = {}
        
        for record in self.wear_history:
            geology = record['geological_type']
            if geology not in geo_analysis:
                geo_analysis[geology] = []
            geo_analysis[geology].append(record)
        
        # Find optimal parameters for each geological type
        for geology, records in geo_analysis.items():
            # Find records with lowest wear rates
            sorted_records = sorted(records, key=lambda x: x['wear_rate'])
            best_10_percent = sorted_records[:max(1, len(sorted_records) // 10)]
            
            # Calculate average optimal parameters
            optimal_speed = sum(r['advance_speed'] for r in best_10_percent) / len(best_10_percent)
            optimal_rpm = sum(r['revolution_rpm'] for r in best_10_percent) / len(best_10_percent)
            optimal_pressure = sum(r['working_pressure'] for r in best_10_percent) / len(best_10_percent)
            avg_wear_rate = sum(r['wear_rate'] for r in best_10_percent) / len(best_10_percent)
            
            self.optimal_parameters[geology] = {
                'advance_speed': round(optimal_speed, 1),
                'revolution_rpm': round(optimal_rpm, 1),
                'working_pressure': round(optimal_pressure, 0),
                'expected_wear_rate': round(avg_wear_rate, 4)
            }
        
        print(f"Analyzed optimal parameters for {len(geo_analysis)} geological types")
        return self.optimal_parameters
    
    def predict_cutter_wear(self, current_conditions):
        """Predict cutter wear for given conditions"""
        
        geology = current_conditions.get('geological_type', 'mixed_ground')
        advance_speed = current_conditions.get('advance_speed', 45)
        rpm = current_conditions.get('revolution_rpm', 8.5)
        earth_pressure = current_conditions.get('earth_pressure', 150)
        total_force = current_conditions.get('total_force', 800)
        current_hours = current_conditions.get('operating_hours', 100)
        current_wear = current_conditions.get('current_wear', 0)
        position = current_conditions.get('cutter_position', 'face_outer')
        
        # Calculate predicted wear rate
        predicted_wear_rate = self.calculate_wear_rate(
            geology, advance_speed, rpm, earth_pressure, total_force, current_hours
        )
        
        # Apply position multiplier
        position_multiplier = self.cutter_positions.get(position, 1.0)
        actual_wear_rate = predicted_wear_rate * position_multiplier
        
        # Predict wear after 24 hours
        predicted_24h_wear = current_wear + (actual_wear_rate * 24)
        
        # Calculate remaining life
        remaining_life = (self.wear_thresholds['replacement_needed'] - current_wear) / actual_wear_rate
        
        return {
            'current_wear': current_wear,
            'predicted_24h_wear': round(predicted_24h_wear, 3),
            'wear_rate_per_hour': round(actual_wear_rate, 4),
            'wear_condition': self.classify_wear_condition(predicted_24h_wear),
            'remaining_life_hours': max(0, round(remaining_life, 1)),
            'replacement_recommended': predicted_24h_wear >= self.wear_thresholds['heavy_wear']
        }
    
    def optimize_boring_parameters(self, geological_conditions):
        """Optimize boring parameters for geological conditions"""
        
        geology = geological_conditions.get('geological_type', 'mixed_ground')
        
        if geology in self.optimal_parameters:
            optimal = self.optimal_parameters[geology]
        else:
            # Use default parameters if geology not in database
            optimal = {
                'advance_speed': 45.0,
                'revolution_rpm': 8.5,
                'working_pressure': 180,
                'expected_wear_rate': 0.15
            }
        
        # Add geological-specific recommendations
        recommendations = self.get_geological_recommendations(geology)
        
        return {
            'optimal_advance_speed': optimal['advance_speed'],
            'optimal_revolution_rpm': optimal['revolution_rpm'],
            'optimal_working_pressure': optimal['working_pressure'],
            'expected_wear_rate': optimal.get('expected_wear_rate', 0.15),
            'geological_recommendations': recommendations
        }
    
    def get_geological_recommendations(self, geology):
        """Get specific recommendations for geological conditions"""
        
        recommendations = []
        
        if geology in ['hard_rock', 'medium_rock']:
            recommendations.extend([
                "Use carbide-tipped cutters for hard rock conditions",
                "Reduce advance speed to minimize cutter wear",
                "Increase cutting pressure for better penetration",
                "Monitor vibration levels closely"
            ])
        elif geology in ['gravel', 'dense_sand']:
            recommendations.extend([
                "Monitor cutter wear closely in abrasive conditions",
                "Consider disc cutter configuration optimization",
                "Maintain consistent advance speed",
                "Implement frequent visual inspections"
            ])
        elif geology in ['soft_clay', 'loose_sand']:
            recommendations.extend([
                "Optimize advance speed for maximum productivity",
                "Monitor for clogging in sticky conditions",
                "Adjust working pressure for soil conditions",
                "Consider soil conditioning if needed"
            ])
        else:  # mixed_ground, weathered_rock
            recommendations.extend([
                "Prepare for variable conditions",
                "Monitor parameters continuously",
                "Be ready to adjust cutting parameters",
                "Implement enhanced wear monitoring"
            ])
        
        return recommendations
    
    def generate_maintenance_schedule(self, cutter_conditions):
        """Generate maintenance schedule based on wear predictions"""
        
        schedule = {
            'immediate_action': [],
            'scheduled_maintenance': [],
            'inspection_due': [],
            'routine_monitoring': []
        }
        
        for conditions in cutter_conditions:
            cutter_id = conditions.get('cutter_id', 'Unknown')
            prediction = self.predict_cutter_wear(conditions)
            
            if prediction['replacement_recommended']:
                schedule['immediate_action'].append({
                    'cutter_id': cutter_id,
                    'action': 'Replace cutter immediately',
                    'current_wear': prediction['current_wear'],
                    'condition': prediction['wear_condition'],
                    'priority': 'HIGH'
                })
            elif prediction['remaining_life_hours'] < 100:
                schedule['scheduled_maintenance'].append({
                    'cutter_id': cutter_id,
                    'action': 'Schedule replacement',
                    'remaining_hours': prediction['remaining_life_hours'],
                    'condition': prediction['wear_condition'],
                    'priority': 'MEDIUM'
                })
            elif prediction['remaining_life_hours'] < 200:
                schedule['inspection_due'].append({
                    'cutter_id': cutter_id,
                    'action': 'Detailed inspection required',
                    'remaining_hours': prediction['remaining_life_hours'],
                    'condition': prediction['wear_condition'],
                    'priority': 'LOW'
                })
            else:
                schedule['routine_monitoring'].append({
                    'cutter_id': cutter_id,
                    'action': 'Continue routine monitoring',
                    'remaining_hours': prediction['remaining_life_hours'],
                    'condition': prediction['wear_condition']
                })
        
        return schedule
    
    def create_analysis_report(self):
        """Generate comprehensive analysis report"""
        
        if not self.wear_history:
            return "No data available for analysis"
        
        report = []
        report.append("="*80)
        report.append("MTBM CUTTER WEAR ANALYSIS REPORT")
        report.append("="*80)
        
        # Overall statistics
        total_cutters = len(set(r['cutter_id'] for r in self.wear_history))
        max_hours = max(r['operating_hours'] for r in self.wear_history)
        max_distance = max(r['tunnel_distance'] for r in self.wear_history)
        
        report.append(f"\nOVERALL STATISTICS:")
        report.append(f"  Total cutters monitored: {total_cutters}")
        report.append(f"  Operating time: {max_hours:.1f} hours")
        report.append(f"  Tunnel distance: {max_distance:.1f}m")
        
        # Wear condition distribution
        wear_conditions = {}
        for record in self.wear_history:
            condition = record['wear_condition']
            wear_conditions[condition] = wear_conditions.get(condition, 0) + 1
        
        report.append(f"\nWEAR CONDITION DISTRIBUTION:")
        total_records = len(self.wear_history)
        for condition, count in wear_conditions.items():
            percentage = (count / total_records) * 100
            report.append(f"  {condition}: {count} ({percentage:.1f}%)")
        
        # Geological impact analysis
        geo_analysis = {}
        for record in self.wear_history:
            geology = record['geological_type']
            if geology not in geo_analysis:
                geo_analysis[geology] = []
            geo_analysis[geology].append(record['wear_rate'])
        
        report.append(f"\nWEAR RATE BY GEOLOGICAL TYPE:")
        for geology, rates in geo_analysis.items():
            avg_rate = sum(rates) / len(rates)
            max_rate = max(rates)
            min_rate = min(rates)
            report.append(f"  {geology}: Avg {avg_rate:.4f}, Range {min_rate:.4f}-{max_rate:.4f} mm/hr")
        
        # Position-based analysis
        position_analysis = {}
        for record in self.wear_history:
            position = record['cutter_position']
            if position not in position_analysis:
                position_analysis[position] = []
            position_analysis[position].append(record['total_wear'])
        
        report.append(f"\nWEAR BY CUTTER POSITION:")
        for position, wear_values in position_analysis.items():
            avg_wear = sum(wear_values) / len(wear_values)
            max_wear = max(wear_values)
            report.append(f"  {position}: Avg {avg_wear:.2f}mm, Max {max_wear:.2f}mm")
        
        # Critical findings
        critical_cutters = [r for r in self.wear_history if r['wear_condition'] in ['heavy_wear', 'replacement_needed']]
        
        report.append(f"\nCRITICAL FINDINGS:")
        if critical_cutters:
            report.append(f"  • {len(critical_cutters)} cutters require immediate attention")
            
            # Group by geological type
            critical_geo = {}
            for cutter in critical_cutters:
                geo = cutter['geological_type']
                critical_geo[geo] = critical_geo.get(geo, 0) + 1
            
            report.append(f"  • Critical cutters by geology:")
            for geo, count in critical_geo.items():
                report.append(f"    - {geo}: {count} cutters")
        else:
            report.append(f"  • No cutters currently require immediate replacement")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        
        # Find highest wear geology
        highest_wear_geo = max(geo_analysis.keys(), 
                              key=lambda g: sum(geo_analysis[g]) / len(geo_analysis[g]))
        avg_highest = sum(geo_analysis[highest_wear_geo]) / len(geo_analysis[highest_wear_geo])
        
        report.append(f"  • Highest wear rates in {highest_wear_geo} ({avg_highest:.4f} mm/hr)")
        report.append(f"  • Consider parameter optimization for {highest_wear_geo} conditions")
        
        if critical_cutters:
            report.append(f"  • Schedule immediate maintenance for {len(critical_cutters)} critical cutters")
        
        # Overall wear rate assessment
        overall_avg_rate = sum(r['wear_rate'] for r in self.wear_history) / len(self.wear_history)
        if overall_avg_rate > 0.2:
            report.append(f"  • High overall wear rate ({overall_avg_rate:.4f} mm/hr) - review operations")
        else:
            report.append(f"  • Overall wear rate acceptable ({overall_avg_rate:.4f} mm/hr)")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def export_analysis_data(self, filename="cutter_wear_analysis.csv"):
        """Export analysis data to CSV file"""
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                if not self.wear_history:
                    print("No data to export")
                    return False
                
                fieldnames = self.wear_history[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in self.wear_history:
                    writer.writerow(record)
                
                print(f"Exported {len(self.wear_history)} records to {filename}")
                return True
                
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False


def main():
    """Main demonstration function"""
    
    print("MTBM CUTTER WEAR PREDICTION & GEOLOGICAL CORRELATION DEMONSTRATION")
    print("="*80)
    
    # Initialize the demo system
    cutter_system = CutterWearDemo()
    
    # Generate comprehensive tunneling data
    print("\n1. GENERATING TUNNELING DATA")
    print("-" * 40)
    tunneling_data = cutter_system.generate_tunneling_data(tunnel_length=500, readings=1000)
    
    # Analyze geological correlations
    print("\n2. GEOLOGICAL CORRELATION ANALYSIS")
    print("-" * 40)
    optimal_params = cutter_system.analyze_geological_correlation()
    
    print("Optimal parameters by geological type:")
    for geology, params in optimal_params.items():
        print(f"  {geology}:")
        print(f"    Speed: {params['advance_speed']} mm/min")
        print(f"    RPM: {params['revolution_rpm']}")
        print(f"    Pressure: {params['working_pressure']} bar")
        print(f"    Expected wear: {params['expected_wear_rate']} mm/hr")
    
    # Demonstrate wear prediction
    print("\n3. CUTTER WEAR PREDICTION EXAMPLES")
    print("-" * 40)
    
    test_conditions = [
        {
            'cutter_id': 'face_center_1',
            'geological_type': 'hard_rock',
            'advance_speed': 35.0,
            'revolution_rpm': 7.5,
            'earth_pressure': 280,
            'total_force': 1400,
            'operating_hours': 150,
            'current_wear': 4.2,
            'cutter_position': 'face_center'
        },
        {
            'cutter_id': 'gauge_2',
            'geological_type': 'dense_sand',
            'advance_speed': 50.0,
            'revolution_rpm': 9.0,
            'earth_pressure': 140,
            'total_force': 700,
            'operating_hours': 200,
            'current_wear': 6.8,
            'cutter_position': 'gauge'
        },
        {
            'cutter_id': 'face_outer_3',
            'geological_type': 'soft_clay',
            'advance_speed': 55.0,
            'revolution_rpm': 8.8,
            'earth_pressure': 80,
            'total_force': 400,
            'operating_hours': 120,
            'current_wear': 1.5,
            'cutter_position': 'face_outer'
        }
    ]
    
    predictions = []
    for i, conditions in enumerate(test_conditions, 1):
        print(f"\nExample {i} - {conditions['cutter_id']} in {conditions['geological_type']}:")
        prediction = cutter_system.predict_cutter_wear(conditions)
        predictions.append(prediction)
        
        print(f"  Current wear: {prediction['current_wear']}mm")
        print(f"  Predicted 24h wear: {prediction['predicted_24h_wear']}mm")
        print(f"  Wear rate: {prediction['wear_rate_per_hour']} mm/hr")
        print(f"  Condition: {prediction['wear_condition']}")
        print(f"  Remaining life: {prediction['remaining_life_hours']} hours")
        print(f"  Replacement needed: {'Yes' if prediction['replacement_recommended'] else 'No'}")
    
    # Demonstrate parameter optimization
    print("\n4. BORING PARAMETER OPTIMIZATION")
    print("-" * 40)
    
    geo_examples = ['hard_rock', 'gravel', 'soft_clay']
    
    for geology in geo_examples:
        print(f"\nOptimization for {geology}:")
        geological_conditions = {'geological_type': geology}
        optimization = cutter_system.optimize_boring_parameters(geological_conditions)
        
        print(f"  Optimal advance speed: {optimization['optimal_advance_speed']} mm/min")
        print(f"  Optimal RPM: {optimization['optimal_revolution_rpm']}")
        print(f"  Optimal pressure: {optimization['optimal_working_pressure']} bar")
        print(f"  Expected wear rate: {optimization['expected_wear_rate']} mm/hr")
        print(f"  Recommendations:")
        for rec in optimization['geological_recommendations']:
            print(f"    • {rec}")
    
    # Generate maintenance schedule
    print("\n5. MAINTENANCE SCHEDULE")
    print("-" * 40)
    
    maintenance_schedule = cutter_system.generate_maintenance_schedule(test_conditions)
    
    for category, items in maintenance_schedule.items():
        if items:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for item in items:
                print(f"  • {item['cutter_id']}: {item['action']}")
                if 'remaining_hours' in item:
                    print(f"    Remaining life: {item['remaining_hours']} hours")
                if 'priority' in item:
                    print(f"    Priority: {item['priority']}")
    
    # Generate comprehensive report
    print("\n6. COMPREHENSIVE ANALYSIS REPORT")
    print("-" * 40)
    report = cutter_system.create_analysis_report()
    print(report)
    
    # Export data
    print("\n7. DATA EXPORT")
    print("-" * 40)
    cutter_system.export_analysis_data("cutter_wear_analysis.csv")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("• cutter_wear_analysis.csv - Complete analysis data")
    print("\nThe framework demonstrates:")
    print("• Cutter wear prediction based on geological conditions")
    print("• Geological correlation analysis for parameter optimization")
    print("• Boring parameter recommendations")
    print("• Predictive maintenance scheduling")
    print("• Comprehensive wear analysis reporting")
    
    return cutter_system


if __name__ == "__main__":
    demo_system = main()