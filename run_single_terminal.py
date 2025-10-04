#!/usr/bin/env python3
"""
Single terminal runner for Smart Fertilizer Recommendation System
Runs both Flask backend and React frontend in one terminal
"""
import subprocess
import threading
import time
import os
import sys
import signal
from pathlib import Path

class ServerManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_flask(self):
        """Start Flask backend server"""
        print("🌱 Starting Flask backend server...")
        try:
            flask_process = subprocess.Popen([
                sys.executable, "app.py"
            ], cwd=Path(__file__).parent)
            self.processes.append(flask_process)
            print("✅ Flask backend started on http://127.0.0.1:5000")
        except Exception as e:
            print(f"❌ Failed to start Flask: {e}")
    
    def start_react(self):
        """Start React frontend server"""
        print("⚛️  Starting React chatbot...")
        try:
            react_dir = Path(__file__).parent / "chatbot_integrated"
            react_process = subprocess.Popen([
                "npm", "run", "dev"
            ], cwd=react_dir, shell=True)
            self.processes.append(react_process)
            print("✅ React chatbot started on http://localhost:3000")
        except Exception as e:
            print(f"❌ Failed to start React: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Shutting down servers...")
        self.running = False
        for process in self.processes:
            try:
                process.terminate()
            except:
                pass
        sys.exit(0)
    
    def run(self):
        """Run both servers"""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🚀 Starting Smart Fertilizer Recommendation System")
        print("=" * 60)
        
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=self.start_flask)
        flask_thread.daemon = True
        flask_thread.start()
        
        # Wait a moment for Flask to start
        time.sleep(2)
        
        # Start React in a separate thread
        react_thread = threading.Thread(target=self.start_react)
        react_thread.daemon = True
        react_thread.start()
        
        # Wait a moment for React to start
        time.sleep(3)
        
        print("\n" + "=" * 60)
        print("🎉 Both servers are running!")
        print("📱 Flask backend: http://127.0.0.1:5000")
        print("🤖 React chatbot: http://localhost:3000")
        print("🔗 Assistant page: http://127.0.0.1:5000/assistant")
        print("\n💡 Press Ctrl+C to stop both servers")
        print("=" * 60)
        
        # Keep the main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.signal_handler(None, None)

if __name__ == "__main__":
    manager = ServerManager()
    manager.run()
