"""
Quick script to check if port 8080 is in use and find the process
"""

import socket
import subprocess
import sys
import os

def check_port(port: int = 8080):
    """Check if a port is in use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        print(f"✗ Port {port} is in use!")
        print("\nTo find the process using this port:")
        if os.name == 'nt':  # Windows
            print(f"  netstat -ano | findstr :{port}")
            print("\nThen kill it with:")
            print(f"  taskkill /PID <PID> /F")
            
            # Try to find it automatically
            try:
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) > 4:
                            pid = parts[-1]
                            print(f"\nFound process ID: {pid}")
                            print(f"Kill with: taskkill /PID {pid} /F")
            except:
                pass
        else:  # Linux/Mac
            print(f"  lsof -i :{port}")
            print("\nThen kill it with:")
            print(f"  kill -9 <PID>")
        return False
    else:
        print(f"✓ Port {port} is available")
        return True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    check_port(port)

