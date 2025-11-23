import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("Testing graph generation...")

# Generate simple test data
np.random.seed(42)
n_samples = 100
timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)]
tunnel_length = np.cumsum(np.random.uniform(0.5, 2.0, n_samples))
deviation = np.cumsum(np.random.normal(0, 2, n_samples))

# Create simple plot
plt.figure(figsize=(10, 6))
plt.plot(timestamps, tunnel_length, label='Tunnel Length (m)', color='blue', linewidth=2)
plt.plot(timestamps, deviation, label='Deviation (mm)', color='red', linewidth=2)
plt.title('MTBM Test Graph - Tunnel Progress and Deviation')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('test_mtbm_graph.png', dpi=300, bbox_inches='tight')
print("✅ Test graph saved: test_mtbm_graph.png")
plt.close()

print("Graph generation test completed!")
