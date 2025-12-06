import os
print("Current directory:", os.getcwd())
print("Files in directory:")
for file in os.listdir('.'):
    print(f"  {file}")

# Test matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\nTesting matplotlib...")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.title('Simple Test Graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    filename = 'simple_test_graph.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    if os.path.exists(filename):
        print(f"✅ Successfully created: {filename}")
        print(f"File size: {os.path.getsize(filename)} bytes")
    else:
        print("❌ Failed to create graph file")
        
except Exception as e:
    print(f"❌ Error with matplotlib: {e}")

print("\nFiles after test:")
for file in os.listdir('.'):
    if file.endswith('.png'):
        print(f"  📊 {file}")
