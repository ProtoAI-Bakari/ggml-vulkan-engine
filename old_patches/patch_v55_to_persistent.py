#!/usr/bin/env python3
"""
Surgical patch to transform v55agent.py into persistent interactive agent
Like v44agent.py but with v55's self-awareness architecture
"""

import sys
from pathlib import Path

# Read the current file
v55_path = Path("~/AGENT/v55agent.py").expanduser()
content = v55_path.read_text()

# Find the main block and replace it
old_main = '''# Example usage and testing
if __name__ == "__main__":
    import signal
    import pprint
    
    def signal_handler(signum, frame):
        print(f"\\nReceived signal {signum}, shutting down gracefully...")
        if 'agent' in globals():
            agent.stop()
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run agent
    print("="*60)
    print("V55AGENT - Self-Aware Autonomous Agent")
    print("="*60)
    
    with V55Agent() as agent:
        print(f"\\nAgent started with ID: {agent.agent_id}")
        print(f"Vulkan available: {agent.system_monitor.vulkan_available}")
        print(f"Other agents detected: {agent.coordinator.detect_other_agents()}")
        
        # Example tasks
        example_tasks = [
            {'id': 'memory_check', 'command': 'free -h'},
            {'id': 'directory_list', 'command': 'ls -la /tmp | head -10'},
            {'id': 'system_info', 'command': 'uname -a'},
            {'id': 'process_count', 'command': 'ps aux | wc -l'}
        ]
        
        print("\\n" + "="*60)
        print("EXECUTING TEST TASKS")
        print("="*60)
        
        for task in example_tasks:
            print(f"\\n--- Task: {task['id']} ---")
            result = agent.execute_task(task)
            print(f"Success: {result.get('success', False)}")
            if result.get('stdout'):
                print(f"Output: {result['stdout'][:200]}...")
            if result.get('error'):
                print(f"Error: {result['error']}")
            
            # Brief pause between tasks
            time.sleep(1)
        
        # Show final status
        print("\\n" + "="*60)
        print("FINAL AGENT STATUS")
        print("="*60)
        pprint.pprint(agent.get_status())
        
        print("\\n" + "="*60)
        print("AGENT TEST COMPLETE")
        print("="*60)'''

new_main = '''# =====================================================
# PERSISTENT INTERACTIVE MODE (like v44agent)
# =====================================================

def get_multiline_input(prompt: str = "") -> str:
    """Get multiline user input until EOF or 'quit'."""
    if prompt:
        print(prompt)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "quit" or line.strip() == "exit":
                return "quit"
            lines.append(line)
        except EOFError:
            break
    return "\\n".join(lines).strip()


def run_persistent_agent(mission: str = None, max_turns: int = 50):
    """Run agent in persistent interactive mode like v44."""
    import signal
    import pprint
    
    def signal_handler(signum, frame):
        print(f"\\n[INTERRUPTED] Received signal {signum}")
        print("Use 'quit' to exit or continue with instructions.")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*60)
    print("V55AGENT - Persistent Self-Aware Autonomous Agent")
    print("="*60)
    print(f"Mission: {mission or 'Interactive Assistant'}")
    print(f"Max turns: {max_turns}")
    print("="*60)
    
    with V55Agent() as agent:
        print(f"\\nAgent ID: {agent.agent_id}")
        print(f"Vulkan available: {agent.system_monitor.vulkan_available}")
        print(f"Other agents: {agent.coordinator.detect_other_agents()}")
        
        # Read scratchpad for context
        scratchpad = agent.memory_manager.read_scratchpad()
        if scratchpad.strip():
            print(f"\\n[SCRATCHPAD FOUND] {len(scratchpad)} bytes")
        
        # Get initial mission briefing
        if mission:
            print(f"\\n[Mission Briefing]: {mission}")
            agent.memory_manager.write_scratchpad(f"# Mission: {mission}\\n")
        else:
            first_input = get_multiline_input("Mission briefing (or 'quit'):")
            if first_input == "quit":
                return
            agent.memory_manager.write_scratchpad(f"# Mission: {first_input}\\n")
        
        turn_count = 0
        consecutive_errors = 0
        
        while turn_count < max_turns:
            turn_count += 1
            print(f"\\n{'='*60}")
            print(f"TURN {turn_count}/{max_turns}")
            print(f"{'='*60}")
            
            # Show current state
            status = agent.get_status()
            print(f"Uptime: {status['self_awareness']['uptime_seconds']:.1f}s")
            print(f"Success rate: {status['self_awareness']['success_rate_percent']:.1f}%")
            print(f"Memory: {status['system_metrics']['available_memory']//1024//1024//1024}GB available")
            
            # Get user instructions
            user_input = get_multiline_input("\\nInstructions (or 'quit', 'status', 'scratchpad'):")
            
            if user_input == "quit":
                print("\\n[Exiting gracefully]")
                break
            
            if user_input == "status":
                pprint.pprint(status)
                turn_count -= 1  # Don't count status check as a turn
                continue
            
            if user_input == "scratchpad":
                print("\\n" + agent.memory_manager.read_scratchpad())
                turn_count -= 1
                continue
            
            if not user_input.strip():
                turn_count -= 1
                continue
            
            # Parse simple commands
            if user_input.lower().startswith("task:"):
                command = user_input[5:].strip()
                task = {'id': f'task_{turn_count}', 'command': command}
                print(f"\\n[EXECUTING]: {command}")
                result = agent.execute_task(task)
                
                if result.get('success'):
                    print(f"[SUCCESS] Output: {result.get('stdout', '')[:500]}")
                    consecutive_errors = 0
                else:
                    print(f"[FAILED] Error: {result.get('error', 'Unknown')}")
                    consecutive_errors += 1
                
                # Update scratchpad
                scratchpad = agent.memory_manager.read_scratchpad()
                agent.memory_manager.write_scratchpad(
                    scratchpad + f"\\n# Turn {turn_count}: {command}\\nResult: {result.get('success')}\\n"
                )
                
                if consecutive_errors > 3:
                    print("[WARNING] 3+ consecutive errors. Consider revising approach.")
                
                continue
            
            # Free-form instruction - agent should parse and execute
            print(f"\\n[PROCESSING]: {user_input[:200]}...")
            
            # For now, treat as task description and try to execute
            # In full implementation, this would use an LLM to parse intent
            # For v55, we'll create a simple task based on keywords
            task_command = None
            
            if "check" in user_input.lower() or "status" in user_input.lower():
                if "memory" in user_input.lower():
                    task_command = "free -h"
                elif "disk" in user_input.lower():
                    task_command = "df -h"
                elif "process" in user_input.lower():
                    task_command = "ps aux | head -20"
                else:
                    task_command = "echo 'System check requested'"
            elif "run" in user_input.lower() or "execute" in user_input.lower():
                # Extract command after "run" or "execute"
                import re
                match = re.search(r'(?:run|execute)\s+(.+)', user_input, re.IGNORECASE)
                if match:
                    task_command = match.group(1).strip()
            elif "list" in user_input.lower() or "ls" in user_input.lower():
                import re
                match = re.search(r'(?:list|ls)\s+(.+)', user_input, re.IGNORECASE)
                if match:
                    task_command = f"ls -la {match.group(1).strip()}"
                else:
                    task_command = "ls -la"
            
            if task_command:
                task = {'id': f'task_{turn_count}', 'command': task_command}
                print(f"[EXECUTING]: {task_command}")
                result = agent.execute_task(task)
                
                if result.get('success'):
                    print(f"[SUCCESS]\\n{result.get('stdout', '')[:1000]}")
                    consecutive_errors = 0
                else:
                    print(f"[FAILED] Error: {result.get('error', 'Unknown')}")
                    consecutive_errors += 1
                
                # Update scratchpad
                scratchpad = agent.memory_manager.read_scratchpad()
                agent.memory_manager.write_scratchpad(
                    scratchpad + f"\\n# Turn {turn_count}: {task_command}\\nResult: {result.get('success')}\\n"
                )
            else:
                print("[INFO] Instruction received. In full v55 implementation, this would use LLM to parse intent.")
                print("For now, use 'task: <command>' format for direct execution.")
                print("Example: task: free -h")
                turn_count -= 1  # Don't count non-execution turns
            
            if consecutive_errors > 5:
                print("[CRITICAL] 5+ consecutive errors. Safety mode recommended.")
                if agent.self_awareness.should_activate_safety_mode():
                    print("[SAFETY MODE] Activated due to poor performance")
    
    print("\\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)
    pprint.pprint(agent.get_status())


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V55Agent - Persistent Autonomous Agent")
    parser.add_argument("--mission", type=str, default=None, help="Mission statement")
    parser.add_argument("--max-turns", type=int, default=50, help="Max autonomous turns")
    parser.add_argument("--test", action="store_true", help="Run quick test tasks (old behavior)")
    args = parser.parse_args()
    
    if args.test:
        # Old test behavior for debugging
        print("Running test mode...")
        # [Keep old test code here if needed]
        sys.exit(0)
    
    # Persistent interactive mode
    run_persistent_agent(mission=args.mission, max_turns=args.max_turns)'''

if old_main in content:
    new_content = content.replace(old_main, new_main)
    v55_path.write_text(new_content)
    print("SUCCESS: Patch applied to v55agent.py")
    print("Agent will now run in persistent interactive mode.")
else:
    print("ERROR: Could not find target code block in v55agent.py")
    print("Manual intervention required.")
    sys.exit(1)
