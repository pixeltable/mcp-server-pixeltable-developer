"""
Persistent Python REPL session for PixelTable MCP server.

Provides a persistent Python execution environment with PixelTable pre-loaded.
"""

import subprocess
import sys
import os
import json
import threading
import queue
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class PersistentPythonREPL:
    """Manages a persistent Python REPL session with PixelTable pre-loaded."""
    
    def __init__(self):
        self.process = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.session_active = False
        self.session_lock = threading.Lock()
        self._initialization_attempted = False
        # Don't initialize immediately - do it lazily on first use
    
    def _initialize_session(self):
        """Initialize the persistent Python session."""
        try:
            # Start Python in interactive mode
            env = os.environ.copy()
            # Ensure we're using the same environment as the MCP server
            
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-i"],  # -u for unbuffered, -i for interactive
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=0  # Unbuffered
            )
            
            # Start background thread to handle output
            self._start_output_handler()
            
            # Pre-load essentials and PixelTable (with timeout)
            self._execute_initialization_code()
            
            self.session_active = True
            logger.info("Persistent Python REPL session initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize REPL session: {e}")
            # Don't raise - let the session be created without REPL
            self.session_active = False
    
    def _start_output_handler(self):
        """Start background thread to collect output from the Python process."""
        def output_collector():
            while self.process and self.process.poll() is None:
                try:
                    line = self.process.stdout.readline()
                    if line:
                        self.output_queue.put(line.rstrip())
                except Exception as e:
                    logger.error(f"Error reading process output: {e}")
                    break
        
        thread = threading.Thread(target=output_collector, daemon=True)
        thread.start()
    
    def _execute_initialization_code(self):
        """Execute initialization code to set up the session."""
        init_code = """
import sys
import os
import json
import inspect
from typing import Any, Dict, List, Optional

# Set up for better output formatting
import pprint
pp = pprint.PrettyPrinter(indent=2, width=80)

# Try to import pixeltable
try:
    import pixeltable as pxt
    print("✓ PixelTable imported successfully")
    print(f"✓ PixelTable version: {pxt.__version__}")
except ImportError as e:
    print(f"⚠ Failed to import PixelTable: {e}")
    pxt = None

# Set up session marker
print("=== PIXELTABLE_REPL_READY ===")
"""
        
        self._send_code(init_code)
        
        # Wait for initialization to complete
        self._wait_for_output("=== PIXELTABLE_REPL_READY ===", timeout=10)
    
    def _send_code(self, code: str):
        """Send code to the Python process."""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("Python process is not running")
        
        try:
            self.process.stdin.write(code + "\n")
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to send code to process: {e}")
            raise
    
    def _wait_for_output(self, marker: str, timeout: int = 5) -> List[str]:
        """Wait for specific output marker and collect all output."""
        output_lines = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                line = self.output_queue.get(timeout=0.1)
                output_lines.append(line)
                if marker in line:
                    break
            except queue.Empty:
                continue
        
        return output_lines
    
    def execute_python(self, code: str, reset: bool = False) -> Dict[str, Any]:
        """
        Execute Python code in the persistent session.
        
        Args:
            code: Python code to execute
            reset: If True, reset the session variables
            
        Returns:
            Dictionary with execution results
        """
        with self.session_lock:
            try:
                # Initialize session if not already done
                if not self._initialization_attempted:
                    self._initialize_session()
                    self._initialization_attempted = True
                
                if reset:
                    self._reset_session()
                
                if not self.session_active:
                    return {
                        "success": False,
                        "error": "REPL session is not active - initialization may have failed",
                        "output": "",
                        "stderr": ""
                    }
                
                # Create a unique marker for this execution
                execution_id = f"EXEC_{int(time.time() * 1000000)}"
                
                # Use exec() approach instead of try/except wrapping
                wrapped_code = f"""
# Execution {execution_id}
import sys
from io import StringIO
import traceback

# Capture stdout
_old_stdout = sys.stdout
_captured_output = StringIO()
sys.stdout = _captured_output

# Prepare for execution
_success = True
_error_msg = None

try:
    # Execute the user code using exec
    exec({repr(code)})
except Exception as e:
    _success = False
    _error_msg = traceback.format_exc()

# Restore stdout
sys.stdout = _old_stdout
_output = _captured_output.getvalue()

# Output results
if _success:
    print(f"=== OUTPUT_{execution_id} ===")
    if _output.strip():
        print(_output.rstrip())
    print(f"=== END_{execution_id} ===")
else:
    print(f"=== ERROR_{execution_id} ===")
    print(_error_msg)
    print(f"=== END_{execution_id} ===")
"""
                
                # Send the code
                self._send_code(wrapped_code)
                
                # Wait for output
                output_lines = self._wait_for_output(f"=== END_{execution_id} ===", timeout=30)
                
                # Parse the output
                return self._parse_execution_output(output_lines, execution_id)
                
            except Exception as e:
                logger.error(f"Error executing Python code: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "output": "",
                    "stderr": ""
                }
    
    def _indent_code(self, code: str, indent: str) -> str:
        """Indent code for wrapping in try/except block."""
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)
    
    def _parse_execution_output(self, output_lines: List[str], execution_id: str) -> Dict[str, Any]:
        """Parse the execution output to extract results and errors."""
        result = {
            "success": True,
            "output": "",
            "stderr": "",
            "error": None
        }
        
        # Find the relevant output section
        in_output = False
        in_error = False
        output_content = []
        error_content = []
        
        for line in output_lines:
            if f"=== OUTPUT_{execution_id} ===" in line:
                in_output = True
                in_error = False
                continue
            elif f"=== ERROR_{execution_id} ===" in line:
                in_error = True
                in_output = False
                result["success"] = False
                continue
            elif f"=== END_{execution_id} ===" in line:
                break
            
            if in_output:
                output_content.append(line)
            elif in_error:
                error_content.append(line)
        
        result["output"] = "\n".join(output_content)
        if error_content:
            result["stderr"] = "\n".join(error_content)
            result["error"] = result["stderr"]
        
        return result
    
    def _reset_session(self):
        """Reset the session by clearing variables."""
        reset_code = """
# Clear all user-defined variables
_to_keep = {'__name__', '__doc__', '__package__', '__loader__', '__spec__', '__builtins__', '__file__'}
_to_keep.update({'sys', 'os', 'json', 'inspect', 'pprint', 'pp', 'pxt'})
_user_vars = [k for k in globals().keys() if not k.startswith('_') and k not in _to_keep]
for _var in _user_vars:
    del globals()[_var]
del _to_keep, _user_vars, _var
print("Session variables cleared")
"""
        self._send_code(reset_code)
        self._wait_for_output("Session variables cleared", timeout=5)
    
    def introspect_function(self, function_path: str) -> Dict[str, Any]:
        """
        Get documentation and signature for a function.
        
        Args:
            function_path: e.g. "pxt.create_table" or "inspect.signature"
            
        Returns:
            Dictionary with function information
        """
        introspect_code = f"""
try:
    import inspect
    
    # Parse the function path
    parts = "{function_path}".split('.')
    obj = globals().get(parts[0])
    if obj is None:
        print("ERROR: Object not found in global namespace")
    else:
        for part in parts[1:]:
            obj = getattr(obj, part, None)
            if obj is None:
                print(f"ERROR: Attribute '{{part}}' not found")
                break
        
        if obj is not None:
            info = {{}}
            try:
                info['signature'] = str(inspect.signature(obj))
            except:
                info['signature'] = "Could not get signature"
            
            try:
                info['docstring'] = inspect.getdoc(obj) or "No docstring available"
            except:
                info['docstring'] = "Could not get docstring"
            
            try:
                info['module'] = getattr(obj, '__module__', 'Unknown')
            except:
                info['module'] = 'Unknown'
            
            try:
                if hasattr(obj, '__call__'):
                    sig = inspect.signature(obj)
                    info['parameters'] = [param.name for param in sig.parameters.values()]
                else:
                    info['parameters'] = []
            except:
                info['parameters'] = []
            
            import json
            print("INTROSPECT_RESULT:")
            print(json.dumps(info, indent=2))
        
except Exception as e:
    print(f"INTROSPECT_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
"""
        
        result = self.execute_python(introspect_code)
        
        if result["success"]:
            output = result["output"]
            if "INTROSPECT_RESULT:" in output:
                try:
                    json_start = output.find("INTROSPECT_RESULT:") + len("INTROSPECT_RESULT:")
                    json_str = output[json_start:].strip()
                    return json.loads(json_str)
                except:
                    pass
        
        return {
            "error": result.get("error", "Failed to introspect function"),
            "signature": "Unknown",
            "docstring": "Could not retrieve documentation",
            "module": "Unknown",
            "parameters": []
        }
    
    def list_available_functions(self, module_name: str = "pxt") -> Dict[str, Any]:
        """List available functions in a module."""
        list_code = f"""
try:
    module = globals().get("{module_name}")
    if module is None:
        print("ERROR: Module not found")
    else:
        functions = [name for name in dir(module) if not name.startswith('_') and callable(getattr(module, name, None))]
        import json
        print("FUNCTIONS_RESULT:")
        print(json.dumps(functions, indent=2))
except Exception as e:
    print(f"FUNCTIONS_ERROR: {{e}}")
"""
        
        result = self.execute_python(list_code)
        
        if result["success"] and "FUNCTIONS_RESULT:" in result["output"]:
            try:
                json_start = result["output"].find("FUNCTIONS_RESULT:") + len("FUNCTIONS_RESULT:")
                json_str = result["output"][json_start:].strip()
                return {"functions": json.loads(json_str)}
            except:
                pass
        
        return {"functions": [], "error": result.get("error", "Failed to list functions")}
    
    def install_package(self, package_name: str) -> Dict[str, Any]:
        """Install a package using uv in the current environment."""
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "stderr": result.stderr,
                "package": package_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "package": package_name
            }
    
    def cleanup(self):
        """Clean up the REPL session."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None
                self.session_active = False


# Global REPL instance
_repl_instance = None


def get_repl_instance() -> PersistentPythonREPL:
    """Get or create the global REPL instance."""
    global _repl_instance
    if _repl_instance is None:
        _repl_instance = PersistentPythonREPL()
    return _repl_instance
