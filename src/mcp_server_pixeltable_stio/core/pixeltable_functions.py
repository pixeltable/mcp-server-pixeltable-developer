"""
Core Pixeltable functions for MCP server.

This module wraps the main Pixeltable API functions for use in the MCP server.
"""

import logging
import json
import subprocess
import sys
import io
import os
from typing import Any, Dict, List, Optional, Union
from functools import wraps

# Import pixeltable modules
try:
    import pixeltable as pxt
except ImportError as e:
    logging.error(f"Failed to import pixeltable: {e}")
    pxt = None

logger = logging.getLogger(__name__)

def suppress_pixeltable_output(func):
    """Decorator to suppress stdout/stderr during PixelTable operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Capture stdout and stderr to prevent JSON corruption
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        captured_errors = io.StringIO()
        
        try:
            sys.stdout = captured_output
            sys.stderr = captured_errors
            
            # Execute the actual function
            result = func(*args, **kwargs)
            
            return result
            
        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Log any captured output for debugging
            captured = captured_output.getvalue()
            captured_err = captured_errors.getvalue()
            if captured.strip():
                logger.debug(f"Suppressed stdout from {func.__name__}: {captured.strip()}")
            if captured_err.strip():
                logger.debug(f"Suppressed stderr from {func.__name__}: {captured_err.strip()}")
    
    return wrapper

def ensure_pixeltable_available():
    """Ensure Pixeltable is available and raise an error if not."""
    if pxt is None:
        raise ValueError("Pixeltable is not available. Please install it first.")

def serialize_result(result: Any) -> Dict[str, Any]:
    """Serialize Pixeltable results for JSON transport."""
    try:
        if hasattr(result, 'to_dict'):
            return {"type": "object", "data": result.to_dict()}
        elif hasattr(result, '__dict__'):
            return {"type": "object", "data": {k: str(v) for k, v in result.__dict__.items()}}
        elif isinstance(result, (str, int, float, bool, type(None))):
            return {"type": "primitive", "data": result}
        elif isinstance(result, (list, tuple)):
            return {"type": "list", "data": [serialize_result(item)["data"] for item in result]}
        elif isinstance(result, dict):
            return {"type": "dict", "data": {k: serialize_result(v)["data"] for k, v in result.items()}}
        else:
            return {"type": "string", "data": str(result)}
    except Exception as e:
        logger.warning(f"Failed to serialize result {type(result)}: {e}")
        return {"type": "string", "data": str(result)}

# Core table management functions

def pixeltable_init(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check Pixeltable initialization status and try to resolve issues."""
    try:
        ensure_pixeltable_available()
        
        # Check if Pixeltable is working by trying to list tables
        import os
        logger.info(f"Checking Pixeltable status with PIXELTABLE_HOME={os.environ.get('PIXELTABLE_HOME')}")
        
        try:
            # Test if Pixeltable is working
            tables = pxt.list_tables()
            return {
                "success": True,
                "message": "Pixeltable is initialized and working",
                "version": pxt.__version__,
                "table_count": len(tables)
            }
        except Exception as e:
            # Try to diagnose and fix the issue
            error_msg = str(e)
            if "Circular env initialization detected" in error_msg:
                # Try to reset the circular initialization flag
                try:
                    logger.info("Attempting to reset Pixeltable's circular initialization flag")
                    
                    # Access Pixeltable's internal Env class and reset the __initializing flag
                    from pixeltable.env import Env
                    from pixeltable.config import Config
                    
                    # Reset the class-level initialization flag
                    if hasattr(Env, '_Env__initializing'):
                        logger.info(f"Current __initializing state: {Env._Env__initializing}")
                        Env._Env__initializing = False
                        logger.info("Reset __initializing flag to False")
                    
                    # Also try to reset the singleton instance if it exists in a bad state
                    if hasattr(Env, '_instance') and Env._instance is not None:
                        logger.info("Found existing Env instance, clearing it")
                        Env._instance = None
                    
                    # Reset Config singleton as well
                    if hasattr(Config, '_Config__instance') and Config._Config__instance is not None:
                        logger.info("Found existing Config instance, clearing it")
                        Config._Config__instance = None
                    
                    # Ensure required environment variables are set
                    os.environ['PIXELTABLE_FILE_CACHE_SIZE_G'] = '100'
                    logger.info("Set PIXELTABLE_FILE_CACHE_SIZE_G=100")
                    
                    # Ensure PIXELTABLE_HOME is properly set before re-initialization
                    current_home = os.environ.get('PIXELTABLE_HOME')
                    logger.info(f"Current PIXELTABLE_HOME: {current_home}")
                    
                    # Force set the environment variable to the expanded path
                    if current_home and current_home.startswith('~'):
                        expanded_home = os.path.expanduser(current_home)
                        os.environ['PIXELTABLE_HOME'] = expanded_home
                        logger.info(f"Expanded PIXELTABLE_HOME from {current_home} to {expanded_home}")
                    elif not current_home:
                        # If not set, use the default expanded path
                        default_home = os.path.expanduser('~/.pixeltable')
                        os.environ['PIXELTABLE_HOME'] = default_home
                        logger.info(f"Set PIXELTABLE_HOME to default: {default_home}")
                    
                    # Now try to initialize again
                    logger.info("Attempting fresh initialization after reset")
                    if config_overrides:
                        pxt.init(**config_overrides)
                    else:
                        pxt.init()
                    
                    # Test again
                    tables = pxt.list_tables()
                    return {
                        "success": True,
                        "message": "Pixeltable recovered from circular initialization after reset",
                        "version": pxt.__version__,
                        "table_count": len(tables)
                    }
                except Exception as e2:
                    logger.error(f"Recovery attempt failed: {e2}")
                    return {
                        "success": False,
                        "message": f"Circular initialization detected and recovery failed: {e2}",
                        "version": pxt.__version__,
                        "original_error": error_msg
                    }
            else:
                return {
                    "success": False,
                    "message": f"Pixeltable is imported but not fully functional: {e}",
                    "version": pxt.__version__
                }
            
    except Exception as e:
        logger.error(f"Error checking Pixeltable: {e}")
        raise ValueError(f"Failed to check Pixeltable: {e}")

def pixeltable_create_table(
    path: str,
    schema: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    source_format: Optional[str] = None,
    schema_overrides: Optional[Dict[str, Any]] = None,
    on_error: str = 'abort',
    primary_key: Optional[Union[str, List[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: str = 'on_write',
    if_exists: str = 'error',
    extra_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a new base table."""
    try:
        ensure_pixeltable_available()
        
        # Redirect all output to avoid JSON parsing issues
        import sys
        import io
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Convert schema strings to Pixeltable types if provided
            if schema:
                converted_schema = {}
                for col_name, col_type in schema.items():
                    if isinstance(col_type, str):
                        # Map string type names to Pixeltable types
                        type_mapping = {
                            'int': pxt.Int,
                            'Int': pxt.Int,
                            'string': pxt.String,
                            'String': pxt.String,
                            'float': pxt.Float,
                            'Float': pxt.Float,
                            'bool': pxt.Bool,
                            'Bool': pxt.Bool,
                            'json': pxt.Json,
                            'Json': pxt.Json,
                            'image': pxt.Image,
                            'Image': pxt.Image,
                            'video': pxt.Video,
                            'Video': pxt.Video,
                            'audio': pxt.Audio,
                            'Audio': pxt.Audio,
                            'document': pxt.Document,
                            'Document': pxt.Document,
                            'timestamp': pxt.Timestamp,
                            'Timestamp': pxt.Timestamp,
                            'date': pxt.Date,
                            'Date': pxt.Date,
                        }
                        converted_schema[col_name] = type_mapping.get(col_type, pxt.String)
                    else:
                        converted_schema[col_name] = col_type
                schema = converted_schema
            
            table = pxt.create_table(
                path,  # Pass path as positional argument
                schema=schema,
                source=source,
                source_format=source_format,
                schema_overrides=schema_overrides,
                on_error=on_error,
                primary_key=primary_key,
                num_retained_versions=num_retained_versions,
                comment=comment,
                media_validation=media_validation,
                if_exists=if_exists,
                extra_args=extra_args
            )
            
            return {
                "success": True,
                "table_path": path
            }
        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
    except Exception as e:
        # Ensure stdout/stderr are restored even on error
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass
            
        return {
            "success": False,
            "error": str(e)
        }

# ================================
# SMART DEPENDENCY MANAGEMENT SYSTEM
# ================================

def check_dependencies(expression: str) -> Dict[str, Any]:
    """Check what dependencies are needed for an expression."""
    missing = []
    available = []
    
    # Normalize expression for checking
    expr_lower = expression.lower()
    
    # Check for YOLOX
    if 'yolox' in expr_lower:
        try:
            from pixeltable.ext.functions import yolox
            available.append('yolox')
        except ImportError:
            missing.append({
                'name': 'yolox',
                'packages': ['torch', 'torchvision', 'pixeltable-yolox'],
                'size': '~2.5GB',
                'time': '5-10 minutes',
                'description': 'YOLO object detection'
            })
    
    # Check for OpenAI
    if any(term in expr_lower for term in ['openai', 'gpt']):
        try:
            from pixeltable.functions import openai
            available.append('openai')
        except ImportError:
            missing.append({
                'name': 'openai',
                'packages': ['openai'],
                'size': '~50MB',
                'time': '1-2 minutes',
                'description': 'OpenAI API integration'
            })
    
    # Check for Hugging Face
    if any(term in expr_lower for term in ['huggingface', 'transformers']):
        try:
            from pixeltable.functions import huggingface
            available.append('huggingface')
        except ImportError:
            missing.append({
                'name': 'huggingface',
                'packages': ['transformers', 'torch'],
                'size': '~1.5GB',
                'time': '3-5 minutes',
                'description': 'Hugging Face transformers'
            })
    
    # Check for Anthropic
    if any(term in expr_lower for term in ['anthropic', 'claude']):
        try:
            from pixeltable.functions import anthropic
            available.append('anthropic')
        except ImportError:
            missing.append({
                'name': 'anthropic',
                'packages': ['anthropic'],
                'size': '~20MB',
                'time': '1 minute',
                'description': 'Anthropic Claude API integration'
            })
    
    # Check for other AI services
    ai_services = {
        'fireworks': {'packages': ['fireworks-ai'], 'description': 'Fireworks AI API'},
        'google': {'packages': ['google-genai'], 'description': 'Google Gemini API'},
        'gemini': {'packages': ['google-genai'], 'description': 'Google Gemini API'},
        'replicate': {'packages': ['replicate'], 'description': 'Replicate API'},
        'mistral': {'packages': ['mistralai'], 'description': 'Mistral AI API'},
        'groq': {'packages': ['groq'], 'description': 'Groq API'},
        'together': {'packages': ['together'], 'description': 'Together AI API'},
    }
    
    for service, info in ai_services.items():
        if service in expr_lower:
            try:
                # Try to import from pixeltable.functions
                module = __import__(f'pixeltable.functions.{service}', fromlist=[service])
                available.append(service)
            except ImportError:
                missing.append({
                    'name': service,
                    'packages': info['packages'],
                    'size': '~20MB',
                    'time': '1 minute',
                    'description': info['description']
                })
    
    # Check for audio/speech processing
    if any(term in expr_lower for term in ['whisper', 'speech', 'audio']):
        try:
            from pixeltable.ext.functions import whisperx
            available.append('whisperx')
        except ImportError:
            missing.append({
                'name': 'whisper',
                'packages': ['openai-whisper'],
                'size': '~100MB',
                'time': '2-3 minutes',
                'description': 'OpenAI Whisper speech recognition'
            })
    
    return {
        'missing': missing,
        'available': available,
        'all_satisfied': len(missing) == 0
    }

def pixeltable_check_dependencies(expression: str) -> Dict[str, Any]:
    """Check what dependencies are needed for an expression."""
    try:
        deps = check_dependencies(expression)
        
        if deps['all_satisfied']:
            return {
                'success': True,
                'message': 'All dependencies satisfied',
                'available': deps['available']
            }
        else:
            return {
                'success': True,
                'dependencies_needed': True,
                'missing': deps['missing'],
                'message': f"Need to install {len(deps['missing'])} dependency group(s)"
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def _check_uv_available() -> bool:
    """Check if uv is available in the system."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def _run_uv_install(package: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run uv pip install with proper error handling."""
    if not _check_uv_available():
        raise RuntimeError("uv is not available in the system")
    
    # Get the current Python interpreter path
    python_path = sys.executable
    
    # Use uv pip install with the specific Python environment
    # This ensures packages are installed where the MCP server is running
    return subprocess.run([
        'uv', 'pip', 'install', '--python', python_path, package
    ], capture_output=True, text=True, timeout=timeout)

def _direct_uv_install(package: str) -> Dict[str, Any]:
    """Install a package directly with uv - helper for common packages."""
    try:
        logger.info(f"Installing {package} directly with uv...")
        
        result = _run_uv_install(package, timeout=300)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Failed to install {package} with uv: {result.stderr}',
                'stdout': result.stdout
            }
        
        return {
            'success': True,
            'message': f'Successfully installed {package} with uv',
            'package': package,
            'method': 'direct_uv'
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Installation of {package} timed out after 5 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Installation of {package} failed: {e}'
        }

def pixeltable_install_yolox() -> Dict[str, Any]:
    """Install YOLOX dependencies for object detection using uv."""
    try:
        logger.info("Installing YOLOX dependencies with uv...")
        
        packages = ['torch', 'torchvision', 'pixeltable-yolox']
        
        for package in packages:
            logger.info(f"Installing {package} with uv...")
            result = _run_uv_install(package, timeout=600)  # 10 min timeout
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'Failed to install {package} with uv: {result.stderr}',
                    'stdout': result.stdout
                }
                
            logger.info(f"Successfully installed {package}")
        
        # Verify installation
        try:
            from pixeltable.ext.functions import yolox
            return {
                'success': True,
                'message': 'YOLOX installed successfully with uv! Object detection is now available.',
                'installed_packages': packages,
                'method': 'uv'
            }
        except ImportError as e:
            return {
                'success': False,
                'error': f'Installation completed but import failed: {e}'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Installation timed out after 10 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Installation failed: {e}'
        }

def pixeltable_install_openai() -> Dict[str, Any]:
    """Install OpenAI dependencies using uv."""
    try:
        logger.info("Installing OpenAI dependencies with uv...")
        
        result = _run_uv_install('openai', timeout=300)  # 5 min timeout
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Failed to install openai with uv: {result.stderr}',
                'stdout': result.stdout
            }
            
        logger.info("Successfully installed openai")
        
        # Verify installation
        try:
            from pixeltable.functions import openai
            return {
                'success': True,
                'message': 'OpenAI installed successfully with uv! Vision and chat functions are now available.',
                'installed_packages': ['openai'],
                'method': 'uv'
            }
        except ImportError as e:
            return {
                'success': False,
                'error': f'Installation completed but import failed: {e}'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Installation timed out after 5 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Installation failed: {e}'
        }

def pixeltable_install_huggingface() -> Dict[str, Any]:
    """Install Hugging Face dependencies using uv."""
    try:
        logger.info("Installing Hugging Face dependencies with uv...")
        
        packages = ['transformers', 'torch']
        
        for package in packages:
            logger.info(f"Installing {package} with uv...")
            result = _run_uv_install(package, timeout=600)  # 10 min timeout
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'Failed to install {package} with uv: {result.stderr}',
                    'stdout': result.stdout
                }
                
            logger.info(f"Successfully installed {package}")
        
        # Verify installation
        try:
            from pixeltable.functions import huggingface
            return {
                'success': True,
                'message': 'Hugging Face installed successfully with uv! Transformers and models are now available.',
                'installed_packages': packages,
                'method': 'uv'
            }
        except ImportError as e:
            return {
                'success': False,
                'error': f'Installation completed but import failed: {e}'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Installation timed out after 10 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Installation failed: {e}'
        }

def pixeltable_install_all_dependencies() -> Dict[str, Any]:
    """Install all available AI/ML dependencies using uv."""
    try:
        logger.info("Starting bulk installation of all AI/ML dependencies with uv...")
        results = {}
        
        # Install YOLOX
        logger.info("Installing YOLOX...")
        yolox_result = pixeltable_install_yolox()
        results['yolox'] = yolox_result
        
        # Install OpenAI
        logger.info("Installing OpenAI...")
        openai_result = pixeltable_install_openai()
        results['openai'] = openai_result
        
        # Install Hugging Face
        logger.info("Installing Hugging Face...")
        hf_result = pixeltable_install_huggingface()
        results['huggingface'] = hf_result
        
        # Check overall success
        success_count = sum(1 for r in results.values() if r.get('success', False))
        total_count = len(results)
        
        # Calculate total packages installed
        all_packages = []
        for result in results.values():
            if result.get('success') and 'installed_packages' in result:
                all_packages.extend(result['installed_packages'])
        
        return {
            'success': success_count > 0,
            'message': f'Installed {success_count}/{total_count} dependency groups with uv',
            'method': 'uv',
            'results': results,
            'fully_successful': success_count == total_count,
            'total_packages': list(set(all_packages)),  # Remove duplicates
            'package_count': len(set(all_packages))
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Bulk installation failed: {e}'
        }

def pixeltable_smart_install(package_hint: str) -> Dict[str, Any]:
    """Smart dependency installer that maps common package names to installation routines.
    
    Args:
        package_hint: Name or hint about what to install (e.g., 'ollama', 'torch', 'yolox', 'openai')
    
    Returns:
        Installation result with success status and details
    """
    try:
        # Normalize the hint
        hint = package_hint.lower().strip()
        
        # Smart mapping of hints to our install functions
        install_map = {
            # YOLOX and object detection
            'yolox': pixeltable_install_yolox,
            'yolo': pixeltable_install_yolox,
            'object_detection': pixeltable_install_yolox,
            'detection': pixeltable_install_yolox,
            'torch': pixeltable_install_yolox,  # torch usually means ML/vision
            'pytorch': pixeltable_install_yolox,
            'torchvision': pixeltable_install_yolox,
            
            # OpenAI
            'openai': pixeltable_install_openai,
            'gpt': pixeltable_install_openai,
            'vision': pixeltable_install_openai,
            'chat': pixeltable_install_openai,
            
            # Hugging Face
            'huggingface': pixeltable_install_huggingface,
            'transformers': pixeltable_install_huggingface,
            'hf': pixeltable_install_huggingface,
            'bert': pixeltable_install_huggingface,
            'llm': pixeltable_install_huggingface,
            'sentence-transformers': lambda: _direct_uv_install('sentence-transformers'),
            'sentence_transformers': lambda: _direct_uv_install('sentence-transformers'),
            
            # AI API Services
            'anthropic': lambda: _direct_uv_install('anthropic'),
            'claude': lambda: _direct_uv_install('anthropic'),
            'fireworks': lambda: _direct_uv_install('fireworks-ai'),
            'fireworks-ai': lambda: _direct_uv_install('fireworks-ai'),
            'google-genai': lambda: _direct_uv_install('google-genai'),
            'google_genai': lambda: _direct_uv_install('google-genai'),
            'gemini': lambda: _direct_uv_install('google-genai'),
            'replicate': lambda: _direct_uv_install('replicate'),
            'mistralai': lambda: _direct_uv_install('mistralai'),
            'mistral': lambda: _direct_uv_install('mistralai'),
            'groq': lambda: _direct_uv_install('groq'),
            'together': lambda: _direct_uv_install('together'),
            
            # Audio/Speech Processing
            'whisper': lambda: _direct_uv_install('openai-whisper'),
            'openai-whisper': lambda: _direct_uv_install('openai-whisper'),
            'whisperx': lambda: _direct_uv_install('whisperx'),
            'speech': lambda: _direct_uv_install('openai-whisper'),
            'audio': lambda: _direct_uv_install('openai-whisper'),
            
            # Common packages that should be installed via direct uv
            'ollama': lambda: _direct_uv_install('ollama'),
            'pillow': lambda: _direct_uv_install('pillow'),
            'numpy': lambda: _direct_uv_install('numpy'),
            'pandas': lambda: _direct_uv_install('pandas'),
        }
        
        # Check if we have a direct mapping
        if hint in install_map:
            logger.info(f"Smart installing {hint} using mapped installer")
            return install_map[hint]()
        
        # If no mapping found, try to install the raw package with uv
        logger.info(f"No mapping found for '{hint}', attempting direct uv install")
        
        if not _check_uv_available():
            return {
                'success': False,
                'error': 'uv is not available and no mapping found for this package'
            }
        
        result = _run_uv_install(hint, timeout=300)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Failed to install {hint} with uv: {result.stderr}',
                'stdout': result.stdout,
                'suggestion': 'Try one of the supported packages: yolox, openai, huggingface'
            }
        
        return {
            'success': True,
            'message': f'Successfully installed {hint} with uv',
            'method': 'direct_uv',
            'package': hint
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Smart install failed: {e}'
        }

def pixeltable_auto_install_for_expression(expression: str) -> Dict[str, Any]:
    """Automatically detect and install dependencies needed for an expression.
    
    Args:
        expression: The expression that needs dependencies
        
    Returns:
        Installation result with details about what was installed
    """
    try:
        # Check what's missing
        deps = check_dependencies(expression)
        
        if deps['all_satisfied']:
            return {
                'success': True,
                'message': 'All dependencies already satisfied',
                'available': deps['available'],
                'installed': []
            }
        
        logger.info(f"Auto-installing {len(deps['missing'])} missing dependencies for expression")
        
        installed = []
        failed = []
        
        for dep in deps['missing']:
            logger.info(f"Installing {dep['name']}...")
            
            # Use our smart installer
            result = pixeltable_smart_install(dep['name'])
            
            if result.get('success', False):
                installed.append(dep['name'])
                logger.info(f"Successfully installed {dep['name']}")
            else:
                failed.append({
                    'name': dep['name'],
                    'error': result.get('error', 'Unknown error')
                })
                logger.error(f"Failed to install {dep['name']}: {result.get('error')}")
        
        if failed:
            return {
                'success': len(installed) > 0,  # Partial success if some worked
                'message': f'Installed {len(installed)}/{len(deps["missing"])} dependencies',
                'installed': installed,
                'failed': failed,
                'partial': True
            }
        else:
            return {
                'success': True,
                'message': f'Successfully installed all {len(installed)} missing dependencies',
                'installed': installed
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Auto-install failed: {e}'
        }

def pixeltable_suggest_install_from_error(error_message: str) -> Dict[str, Any]:
    """Analyze an error message and suggest installations.
    
    Args:
        error_message: The error message to analyze
        
    Returns:
        Suggestions for what to install
    """
    try:
        error_lower = error_message.lower()
        suggestions = []
        
        # Common import error patterns
        if 'no module named' in error_lower:
            # YOLOX and ML packages
            if any(term in error_lower for term in ['yolox', 'torch', 'torchvision']):
                suggestions.append({
                    'package': 'yolox',
                    'function': 'pixeltable_install_yolox',
                    'description': 'Install YOLOX for object detection'
                })
            
            # OpenAI
            if 'openai' in error_lower:
                suggestions.append({
                    'package': 'openai', 
                    'function': 'pixeltable_install_openai',
                    'description': 'Install OpenAI for vision and chat functions'
                })
            
            # Hugging Face
            if any(term in error_lower for term in ['transformers', 'huggingface']):
                suggestions.append({
                    'package': 'huggingface',
                    'function': 'pixeltable_install_huggingface', 
                    'description': 'Install Hugging Face transformers'
                })
            
            # AI API services
            ai_services = {
                'anthropic': 'Install Anthropic Claude API',
                'fireworks': 'Install Fireworks AI API',
                'google-genai': 'Install Google Gemini API',
                'replicate': 'Install Replicate API',
                'mistralai': 'Install Mistral AI API',
                'groq': 'Install Groq API',
                'together': 'Install Together AI API',
            }
            
            for service, description in ai_services.items():
                if service in error_lower or service.replace('-', '_') in error_lower:
                    suggestions.append({
                        'package': service,
                        'function': 'pixeltable_smart_install',
                        'description': description
                    })
            
            # Audio/speech processing
            if any(term in error_lower for term in ['whisper', 'whisperx']):
                suggestions.append({
                    'package': 'whisper',
                    'function': 'pixeltable_smart_install',
                    'description': 'Install Whisper for speech recognition'
                })
                
            # Extract package name if possible
            import re
            match = re.search(r"no module named ['\"]([^'\"]+)['\"]?", error_lower)
            if match:
                package_name = match.group(1)
                suggestions.append({
                    'package': package_name,
                    'function': 'pixeltable_smart_install',
                    'description': f'Try smart install for {package_name}'
                })
        
        if suggestions:
            return {
                'success': True,
                'suggestions': suggestions,
                'message': f'Found {len(suggestions)} installation suggestion(s)'
            }
        else:
            return {
                'success': True,
                'suggestions': [],
                'message': 'No installation suggestions found for this error'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to analyze error: {e}'
        }

def pixeltable_system_diagnostics() -> Dict[str, Any]:
    """Get system diagnostics for Pixeltable and dependencies."""
    try:
        diagnostics = {
            'pixeltable_version': None,
            'uv_available': False,
            'uv_version': None,
            'dependencies': {
                'yolox': False,
                'openai': False, 
                'huggingface': False
            },
            'system_info': {}
        }
        
        # Check Pixeltable version
        try:
            import pixeltable as pxt
            diagnostics['pixeltable_version'] = pxt.__version__
        except ImportError:
            diagnostics['pixeltable_version'] = 'Not installed'
        
        # Check uv availability and version
        if _check_uv_available():
            diagnostics['uv_available'] = True
            try:
                result = subprocess.run(['uv', '--version'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    diagnostics['uv_version'] = result.stdout.strip()
            except:
                pass
        
        # Check AI dependencies
        try:
            from pixeltable.ext.functions import yolox
            diagnostics['dependencies']['yolox'] = True
        except ImportError:
            pass
            
        try:
            from pixeltable.functions import openai
            diagnostics['dependencies']['openai'] = True
        except ImportError:
            pass
            
        try:
            from pixeltable.functions import huggingface
            diagnostics['dependencies']['huggingface'] = True
        except ImportError:
            pass
        
        # System info
        import platform
        diagnostics['system_info'] = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }
        
        return {
            'success': True,
            'diagnostics': diagnostics
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def pixeltable_get_table(path: str) -> Dict[str, Any]:
    """Get a handle to an existing table, view, or snapshot."""
    try:
        ensure_pixeltable_available()
        table = pxt.get_table(path)
        
        return {
            "success": True,
            "message": f"Table '{path}' retrieved successfully",
            "table_path": str(table._path()),
            "table_info": serialize_result(table)
        }
    except Exception as e:
        logger.error(f"Error getting table: {e}")
        raise ValueError(f"Failed to get table: {e}")

@suppress_pixeltable_output
def pixeltable_list_tables(dir_path: str = '', recursive: bool = True) -> Dict[str, Any]:
    """List tables in a directory."""
    try:
        ensure_pixeltable_available()
        tables = pxt.list_tables(dir_path, recursive=recursive)
        
        return {
            "success": True,
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise ValueError(f"Failed to list tables: {e}")

def pixeltable_drop_table(
    table: str, 
    force: bool = False, 
    if_not_exists: str = 'error'
) -> Dict[str, Any]:
    """Drop a table, view, or snapshot."""
    try:
        ensure_pixeltable_available()
        pxt.drop_table(table, force=force, if_not_exists=if_not_exists)
        
        return {
            "success": True,
            "message": f"Table '{table}' dropped successfully"
        }
    except Exception as e:
        logger.error(f"Error dropping table: {e}")
        raise ValueError(f"Failed to drop table: {e}")

def pixeltable_create_view(
    path: str,
    base_table_path: str,
    additional_columns: Optional[Dict[str, Any]] = None,
    is_snapshot: bool = False,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: str = 'on_write',
    if_exists: str = 'error'
) -> Dict[str, Any]:
    """Create a view of an existing table."""
    try:
        ensure_pixeltable_available()
        
        # Redirect all output to avoid JSON parsing issues
        import sys
        import io
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            base_table = pxt.get_table(base_table_path)
            
            view = pxt.create_view(
                path=path,
                base=base_table,
                additional_columns=additional_columns,
                is_snapshot=is_snapshot,
                num_retained_versions=num_retained_versions,
                comment=comment,
                media_validation=media_validation,
                if_exists=if_exists
            )
            
            return {
                "success": True,
                "view_path": path
            }
        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
    except Exception as e:
        # Ensure stdout/stderr are restored even on error
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass
            
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_create_snapshot(
    path: str,
    base_table_path: str,
    additional_columns: Optional[Dict[str, Any]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: str = 'on_write',
    if_exists: str = 'error'
) -> Dict[str, Any]:
    """Create a snapshot of an existing table."""
    try:
        ensure_pixeltable_available()
        base_table = pxt.get_table(base_table_path)
        
        snapshot = pxt.create_snapshot(
            path_str=path,
            base=base_table,
            additional_columns=additional_columns,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=media_validation,
            if_exists=if_exists
        )
        
        return {
            "success": True,
            "message": f"Snapshot '{path}' created successfully",
            "snapshot_path": str(snapshot._path()) if snapshot else path
        }
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}")
        raise ValueError(f"Failed to create snapshot: {e}")

# Directory management functions

def pixeltable_create_dir(
    path: str,
    if_exists: str = 'error',
    parents: bool = False
) -> Dict[str, Any]:
    """Create a directory."""
    try:
        ensure_pixeltable_available()
        directory = pxt.create_dir(path, if_exists=if_exists, parents=parents)
        
        return {
            "success": True,
            "message": f"Directory '{path}' created successfully",
            "directory_path": path
        }
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        raise ValueError(f"Failed to create directory: {e}")

def pixeltable_drop_dir(
    path: str,
    force: bool = False,
    if_not_exists: str = 'error'
) -> Dict[str, Any]:
    """Remove a directory."""
    try:
        ensure_pixeltable_available()
        pxt.drop_dir(path, force=force, if_not_exists=if_not_exists)
        
        return {
            "success": True,
            "message": f"Directory '{path}' dropped successfully"
        }
    except Exception as e:
        logger.error(f"Error dropping directory: {e}")
        raise ValueError(f"Failed to drop directory: {e}")

def pixeltable_list_dirs(path: str = '', recursive: bool = True) -> Dict[str, Any]:
    """List directories in a directory."""
    try:
        ensure_pixeltable_available()
        directories = pxt.list_dirs(path, recursive=recursive)
        
        return {
            "success": True,
            "directories": directories,
            "count": len(directories)
        }
    except Exception as e:
        logger.error(f"Error listing directories: {e}")
        raise ValueError(f"Failed to list directories: {e}")

def pixeltable_ls(path: str = '') -> Dict[str, Any]:
    """List the contents of a Pixeltable directory."""
    try:
        ensure_pixeltable_available()
        df = pxt.ls(path)
        
        # Convert DataFrame to dict for JSON serialization
        result = df.to_dict('records')
        
        return {
            "success": True,
            "contents": result,
            "count": len(result)
        }
    except Exception as e:
        logger.error(f"Error listing directory contents: {e}")
        raise ValueError(f"Failed to list directory contents: {e}")

def pixeltable_move(path: str, new_path: str) -> Dict[str, Any]:
    """Move a schema object to a new directory and/or rename it."""
    try:
        ensure_pixeltable_available()
        pxt.move(path, new_path)
        
        return {
            "success": True,
            "message": f"Moved '{path}' to '{new_path}'"
        }
    except Exception as e:
        logger.error(f"Error moving object: {e}")
        raise ValueError(f"Failed to move object: {e}")

# Function and UDF management

def pixeltable_list_functions() -> Dict[str, Any]:
    """List all registered functions."""
    try:
        ensure_pixeltable_available()
        styled_df = pxt.list_functions()
        
        # Extract the underlying DataFrame
        df = styled_df.data
        
        # Convert to records for JSON serialization
        functions = df.to_dict('records')
        
        return {
            "success": True,
            "functions": functions,
            "count": len(functions)
        }
    except Exception as e:
        logger.error(f"Error listing functions: {e}")
        raise ValueError(f"Failed to list functions: {e}")

# Configuration and utility functions

def pixeltable_configure_logging(
    to_stdout: Optional[bool] = None,
    level: Optional[int] = None,
    add: Optional[str] = None,
    remove: Optional[str] = None
) -> Dict[str, Any]:
    """Configure logging."""
    try:
        ensure_pixeltable_available()
        pxt.configure_logging(to_stdout=to_stdout, level=level, add=add, remove=remove)
        
        return {
            "success": True,
            "message": "Logging configured successfully"
        }
    except Exception as e:
        logger.error(f"Error configuring logging: {e}")
        raise ValueError(f"Failed to configure logging: {e}")

# Type system functions

def pixeltable_get_types() -> Dict[str, Any]:
    """Get available Pixeltable data types."""
    try:
        ensure_pixeltable_available()
        
        types_info = {
            "basic_types": {
                "Int": "Integer type",
                "Float": "Floating point type", 
                "String": "String/text type",
                "Bool": "Boolean type",
                "Json": "JSON object type"
            },
            "temporal_types": {
                "Timestamp": "Timestamp type",
                "Date": "Date type"
            },
            "media_types": {
                "Image": "Image file type",
                "Video": "Video file type", 
                "Audio": "Audio file type",
                "Document": "Document file type"
            },
            "complex_types": {
                "Array": "Array type (requires element type)"
            }
        }
        
        return {
            "success": True,
            "types": types_info
        }
    except Exception as e:
        logger.error(f"Error getting types: {e}")
        raise ValueError(f"Failed to get types: {e}")

# Extended table operations

def pixeltable_create_replica(destination: str, source: str) -> Dict[str, Any]:
    """Create a replica of a table."""
    try:
        ensure_pixeltable_available()
        
        # Get source table if it's a local path
        if not source.startswith('pxt://'):
            source_table = pxt.get_table(source)
            result = pxt.create_replica(destination, source_table)
        else:
            result = pxt.create_replica(destination, source)
        
        return {
            "success": True,
            "message": f"Replica created from '{source}' to '{destination}'"
        }
    except Exception as e:
        logger.error(f"Error creating replica: {e}")
        raise ValueError(f"Failed to create replica: {e}")

# Query and expression functions

def pixeltable_query_table(table_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Execute a simple query on a table."""
    try:
        ensure_pixeltable_available()
        
        table = pxt.get_table(table_path)
        
        # Get table data as DataFrame
        df = table.select()
        if limit:
            df = df.limit(limit)
        
        # Convert to dict for JSON serialization
        # Note: This is simplified - actual implementation would handle
        # complex data types and large datasets differently
        result_data = df.collect()
        
        return {
            "success": True,
            "data": serialize_result(result_data)["data"],
            "row_count": len(result_data) if isinstance(result_data, list) else 1
        }
    except Exception as e:
        logger.error(f"Error querying table: {e}")
        raise ValueError(f"Failed to query table: {e}")

def pixeltable_get_table_schema(table_path: str) -> Dict[str, Any]:
    """Get the schema of a table."""
    try:
        ensure_pixeltable_available()
        
        table = pxt.get_table(table_path)
        
        # Get column information
        columns = []
        for col in table._tbl_version_path.columns():
            columns.append({
                "name": col.name,
                "type": str(col.col_type),
                "nullable": col.col_type.nullable
            })
        
        return {
            "success": True,
            "table_path": table_path,
            "columns": columns,
            "column_count": len(columns)
        }
    except Exception as e:
        logger.error(f"Error getting table schema: {e}")
        raise ValueError(f"Failed to get table schema: {e}")

# Additional utilities

@suppress_pixeltable_output
def pixeltable_get_version() -> Dict[str, Any]:
    """Get Pixeltable version information."""
    try:
        ensure_pixeltable_available()
        
        return {
            "success": True,
            "pixeltable_version": pxt.__version__,
            "mcp_server_version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Error getting version: {e}")
        raise ValueError(f"Failed to get version: {e}")

def pixeltable_insert_data(table_path: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Insert data into a table."""
    try:
        ensure_pixeltable_available()
        
        # Redirect all output to avoid JSON parsing issues
        import sys
        import io
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            table = pxt.get_table(table_path)
            table.insert(data)
            
            # Simple, clean response
            return {
                "success": True,
                "rows_inserted": len(data)
            }
        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
    except Exception as e:
        # Ensure stdout/stderr are restored even on error
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass
            
        return {
            "success": False,
            "error": str(e)
        }

# =================
# HIGH PRIORITY MISSING FUNCTIONS
# =================

def pixeltable_query(table_path: str, limit: Optional[int] = None, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generic query interface for PixelTable.
    
    This is the universal query function that provides flexible querying capabilities.
    
    Args:
        table_path: Path to the table to query
        limit: Maximum number of rows to return
        columns: List of columns to select (if None, selects all)
        
    Returns:
        Query result with success status and data
    """
    try:
        ensure_pixeltable_available()
        
        # Get the table
        table = pxt.get_table(table_path)
        
        # Build the query
        if columns:
            # Select specific columns
            result = table.select(*columns)
        else:
            # Select all columns
            result = table.select()
        
        # Apply limit if specified
        if limit:
            result = result.limit(limit)
            
        # Execute and return results
        data = result.collect()
        
        return {
            "success": True,
            "data": serialize_result(data)["data"],
            "row_count": len(data) if isinstance(data, list) else 1,
            "table_path": table_path,
            "query_info": {
                "columns": columns,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error in pixeltable_query: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_create_udf(function_code: str, function_name: str, kwargs: str = "{}") -> Dict[str, Any]:
    """Create a User Defined Function from code.
    
    Allows dynamic creation of custom functions that can be used
    in computed columns and other PixelTable operations.
    
    Args:
        function_code: Python code for the function
        function_name: Name for the UDF
        kwargs: Additional parameters for UDF creation (JSON string)
        
    Returns:
        Success status and UDF information
    """
    try:
        ensure_pixeltable_available()
        
        # Parse kwargs from JSON string
        import json
        try:
            parsed_kwargs = json.loads(kwargs)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON in kwargs: {e}"
            }
        
        # Create a safe execution environment
        exec_globals = {
            'pxt': pxt,
            'pixeltable': pxt,
            '__builtins__': __builtins__,
        }
        
        # Import commonly used modules
        try:
            import numpy as np
            exec_globals['np'] = np
            exec_globals['numpy'] = np
        except ImportError:
            pass
            
        try:
            from PIL import Image
            exec_globals['Image'] = Image
        except ImportError:
            pass
        
        # Execute the function code
        exec(function_code, exec_globals)
        
        # Check if function was created
        if function_name not in exec_globals:
            return {
                "success": False,
                "error": f"Function '{function_name}' was not defined in the provided code"
            }
        
        created_function = exec_globals[function_name]
        
        # Create the UDF using PixelTable's udf decorator
        udf_func = pxt.udf(created_function, **parsed_kwargs)
        
        return {
            "success": True,
            "message": f"UDF '{function_name}' created successfully",
            "function_name": function_name,
            "udf_info": {
                "name": function_name,
                "type": "user_defined_function",
                "callable": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating UDF: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_create_array(elements: list, kwargs: str = "{}") -> Dict[str, Any]:
    """Create array expressions for PixelTable.
    
    Useful for creating complex data structures and expressions
    that can be used in queries and computed columns.
    
    Args:
        elements: List of elements for the array
        **kwargs: Additional parameters for array creation
        
    Returns:
        Array expression result
    """
    try:
        ensure_pixeltable_available()
        
        # Create PixelTable array expression
        array_expr = pxt.Array(elements)
        
        return {
            "success": True,
            "message": f"Array created with {len(elements)} elements",
            "array_info": {
                "length": len(elements),
                "type": "pixeltable_array",
                "elements": elements[:5] if len(elements) > 5 else elements,  # Show first 5 for preview
                "truncated": len(elements) > 5
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating array: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_create_tools(udfs: str, kwargs: str = "{}") -> Dict[str, Any]:
    """Create tools collection for LLM integration.
    
    Wraps UDFs for use with language models and tool-calling APIs.
    Enables integration between PixelTable functions and AI models.
    
    Args:
        *udfs: UDF functions to wrap as tools
        **kwargs: Additional parameters for tool creation
        
    Returns:
        Tools collection information
    """
    try:
        ensure_pixeltable_available()
        
        tools = []
        for udf in udfs:
            if callable(udf):
                tool_info = {
                    "name": getattr(udf, '__name__', 'unknown'),
                    "type": "udf_tool",
                    "callable": True,
                    "function": udf
                }
                tools.append(tool_info)
            else:
                return {
                    "success": False,
                    "error": f"Invalid UDF provided: {udf} is not callable"
                }
        
        return {
            "success": True,
            "message": f"Created tools collection with {len(tools)} tools",
            "tools": tools,
            "tool_count": len(tools)
        }
        
    except Exception as e:
        logger.error(f"Error creating tools: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_connect_mcp(url: str, kwargs: str = "{}") -> Dict[str, Any]:
    """Connect to external MCP server and import functions.
    
    This enables research dataset sharing and function import capability.
    Can be used to pull in functions from academic papers,
    other research groups, or external AI services.
    
    Args:
        url: URL of the MCP server to connect to
        **kwargs: Additional connection parameters
        
    Returns:
        Connection status and available functions
    """
    try:
        ensure_pixeltable_available()
        
        # This is a placeholder implementation
        # In a real implementation, this would:
        # 1. Connect to the MCP server at the given URL
        # 2. Discover available functions
        # 3. Import and register them in PixelTable
        
        return {
            "success": False,
            "error": "MCP connection not yet implemented",
            "message": "This feature is planned for future release",
            "url": url,
            "status": "not_implemented"
        }
        
    except Exception as e:
        logger.error(f"Error connecting to MCP: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# =================
# DATA TYPE HELPERS
# =================

def pixeltable_create_image_type(kwargs: str = "{}") -> Dict[str, Any]:
    """Return pxt.Image type for schema definition."""
    try:
        ensure_pixeltable_available()
        return {
            "success": True,
            "type": "Image",
            "type_object": pxt.Image,
            "description": "PixelTable Image type for storing image data"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def pixeltable_create_video_type(kwargs: str = "{}") -> Dict[str, Any]:
    """Return pxt.Video type for schema definition."""
    try:
        ensure_pixeltable_available()
        return {
            "success": True,
            "type": "Video",
            "type_object": pxt.Video,
            "description": "PixelTable Video type for storing video data"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def pixeltable_create_audio_type(kwargs: str = "{}") -> Dict[str, Any]:
    """Return pxt.Audio type for schema definition."""
    try:
        ensure_pixeltable_available()
        return {
            "success": True,
            "type": "Audio",
            "type_object": pxt.Audio,
            "description": "PixelTable Audio type for storing audio data"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def pixeltable_create_array_type(element_type=None, kwargs: str = "{}") -> Dict[str, Any]:
    """Return pxt.Array type for schema definition."""
    try:
        ensure_pixeltable_available()
        if element_type:
            array_type = pxt.Array(element_type)
        else:
            array_type = pxt.Array
        return {
            "success": True,
            "type": "Array",
            "type_object": array_type,
            "description": "PixelTable Array type for storing array data",
            "element_type": element_type
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def pixeltable_create_json_type(kwargs: str = "{}") -> Dict[str, Any]:
    """Return pxt.Json type for schema definition."""
    try:
        ensure_pixeltable_available()
        return {
            "success": True,
            "type": "Json",
            "type_object": pxt.Json,
            "description": "PixelTable Json type for storing JSON data"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# =================
# EXISTING FUNCTIONS
# =================

def pixeltable_add_computed_column(
    table_path: str,
    column_name: str,
    expression: str,
    if_exists: str = 'error',
    auto_install: bool = False
) -> Dict[str, Any]:
    """Add a computed column to an existing table with smart dependency management.
    
    Args:
        table_path: Path to the table
        column_name: Name of the new computed column
        expression: Python expression string defining the computation
        if_exists: What to do if column exists ('error', 'replace', 'ignore')
        auto_install: Whether to automatically install missing dependencies
    
    Example expressions:
        - "yolox.yolox(table.image, model_id='yolox_s', threshold=0.5)"
        - "openai.vision('Describe this image', table.image, model='gpt-4o-mini')"
        - "image.width(table.image)"
        - "image.height(table.image)"
    """
    try:
        ensure_pixeltable_available()
        
        # Check dependencies first
        deps = check_dependencies(expression)
        
        if not deps['all_satisfied'] and not auto_install:
            missing_info = []
            for dep in deps['missing']:
                missing_info.append(f"• {dep['name']}: {dep['description']} ({dep['size']}, {dep['time']})")
            
            suggestion = "\n".join([
                "Install missing dependencies first:",
                *[f"  pixeltable_install_{dep['name']}()" for dep in deps['missing']],
                "",
                "Or run with auto_install=True to install automatically"
            ])
            
            return {
                "success": False,
                "error": "Missing dependencies",
                "missing_dependencies": deps['missing'],
                "details": "\n".join(missing_info),
                "suggestion": suggestion
            }
        
        # Auto-install if requested
        if not deps['all_satisfied'] and auto_install:
            logger.info("Auto-installing missing dependencies...")
            install_result = pixeltable_auto_install_for_expression(expression)
            
            if not install_result.get('success', False):
                return {
                    "success": False,
                    "error": "Failed to auto-install dependencies",
                    "install_error": install_result.get('error', 'Unknown error'),
                    "failed_dependencies": install_result.get('failed', [])
                }
        
        # Redirect all output to avoid JSON parsing issues
        import sys
        import io
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Get the table
            table = pxt.get_table(table_path)
            
            # Create evaluation context with table and common imports
            eval_context = {
                'table': table,
                'pxt': pxt,
                'pixeltable': pxt,
            }
            
            # Import commonly used function modules into context
            try:
                from pixeltable.ext.functions import yolox
                eval_context['yolox'] = yolox
            except ImportError:
                pass
                
            try:
                from pixeltable.functions import openai, image, string, math
                eval_context.update({
                    'openai': openai,
                    'image': image, 
                    'string': string,
                    'math': math
                })
            except ImportError:
                pass
                
            try:
                from pixeltable.functions import huggingface
                eval_context['huggingface'] = huggingface
            except ImportError:
                pass
            
            # Evaluate the expression to get the computed column definition
            try:
                computed_expr = eval(expression, eval_context)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to evaluate expression '{expression}': {e}"
                }
            
            # Add the computed column to the table
            kwargs = {column_name: computed_expr}
            if if_exists != 'error':
                kwargs['if_exists'] = if_exists
                
            table.add_computed_column(**kwargs)
            
            # Get updated schema to confirm
            columns = []
            for col in table._tbl_version_path.columns():
                columns.append({
                    "name": col.name,
                    "type": str(col.col_type),
                    "nullable": col.col_type.nullable
                })
            
            return {
                "success": True,
                "message": f"Computed column '{column_name}' added successfully",
                "table_path": table_path,
                "column_name": column_name,
                "expression": expression,
                "dependencies_used": deps['available'],
                "updated_schema": columns
            }
            
        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
    except Exception as e:
        # Ensure stdout/stderr are restored even on error
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass
            
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_set_datastore(path: str) -> Dict[str, Any]:
    """
    Set the Pixeltable datastore path and switch to it.

    Args:
        path: Path to the datastore directory

    Returns:
        Dict with success status and tables in the new datastore
    """
    try:
        from mcp_server_pixeltable_stio.core.config import set_datastore_path
        import pixeltable as pxt
        import os

        # Expand the path
        expanded_path = os.path.expanduser(path)

        # Create directory if it doesn't exist
        if not os.path.exists(expanded_path):
            os.makedirs(expanded_path, exist_ok=True)

        # Update config file
        set_datastore_path(expanded_path)

        # Set environment variable
        os.environ['PIXELTABLE_HOME'] = expanded_path

        # Reinitialize Pixeltable
        pxt.init()

        # Get tables in the new datastore
        tables = pxt.list_tables()

        return {
            "success": True,
            "message": f"Switched to datastore: {expanded_path}",
            "path": expanded_path,
            "tables": tables,
            "table_count": len(tables)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_get_datastore() -> Dict[str, Any]:
    """
    Get the current Pixeltable datastore path configuration.
    
    Returns information about all configuration sources and the effective path.
    
    Returns:
        Dict with configuration information
    """
    try:
        from mcp_server_pixeltable_stio.core.config import load_config, get_config_path
        import os
        import pixeltable as pxt

        # Load config
        config = load_config()
        config_path = get_config_path()
        datastore_path = config.get('storage', {}).get('datastore_path', '~/.pixeltable')
        datastore_path = os.path.expanduser(datastore_path)

        # Check what's currently in use by PIXELTABLE_HOME
        env_path = os.environ.get('PIXELTABLE_HOME')
        currently_active = env_path if env_path else datastore_path

        # Check if the path exists
        exists = os.path.exists(currently_active)

        # Try to get tables from currently active datastore
        try:
            tables = pxt.list_tables()
        except:
            tables = []

        return {
            "success": True,
            "config_file": config_path,
            "configured_path": datastore_path,
            "currently_active": currently_active,
            "exists": exists,
            "tables": tables,
            "table_count": len(tables),
            "cache_size_gb": config.get('cache', {}).get('file_cache_size_gb', 10)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_get_help() -> Dict[str, Any]:
    """
    Get comprehensive help and overview of Pixeltable concepts and workflows.
    
    Returns a structured guide to understanding Pixeltable's core concepts,
    typical workflows, and best practices for multimodal AI applications.
    
    Returns:
        Dict containing help sections and examples
    """
    return {
        "success": True,
        "overview": {
            "what_is_pixeltable": (
                "Pixeltable is a Python framework for multimodal AI applications that provides "
                "incremental storage, transformation, indexing, and orchestration of data. "
                "It's built on PostgreSQL but extends it with native support for images, video, "
                "audio, and documents alongside structured data."
            ),
            "key_innovation": (
                "Unlike traditional databases, Pixeltable uses 'computed columns' that automatically "
                "run AI models and transformations on your data. When you add new data, all dependent "
                "computations update automatically - this is called 'incremental computation'."
            ),
            "architecture": {
                "storage": "Data stored in ~/.pixeltable with PostgreSQL for metadata and flat files for media",
                "computation": "Declarative computed columns that auto-update when data changes",
                "versioning": "Automatic versioning and time-travel queries for all data changes"
            }
        },
        
        "core_concepts": {
            "tables": (
                "Tables store your multimodal data. Unlike SQL tables, they natively handle "
                "images, video, audio, and documents. Create with pixeltable_create_table()."
            ),
            "computed_columns": (
                "Columns that automatically compute values using Python expressions or AI models. "
                "Example: Add object detection that runs on every image automatically. "
                "Use pixeltable_add_computed_column()."
            ),
            "views": (
                "Filtered or transformed views of tables. Like SQL views but with full "
                "multimodal support. Changes to base table automatically propagate."
            ),
            "snapshots": (
                "Immutable copies of a table at a point in time. Useful for reproducibility "
                "and comparing model outputs across versions."
            ),
            "incremental_updates": (
                "When you add data or change a computed column, Pixeltable only recomputes "
                "what's necessary, saving time and compute costs."
            )
        },
        
        "data_types": {
            "multimodal": ["pxt.Image", "pxt.Video", "pxt.Audio", "pxt.Document"],
            "structured": ["String", "Int", "Float", "Bool", "Json", "Array"],
            "special": ["Embeddings (for vector search)", "File paths and URLs"]
        },
        
        "typical_workflows": {
            "1_basic_flow": [
                "Create table: pixeltable_create_table() with schema",
                "Insert data: pixeltable_insert_data() with images/videos/text",
                "Add AI: pixeltable_add_computed_column() with model inference",
                "Query: pixeltable_query_table() to get results"
            ],
            "2_computer_vision": [
                "Create table with pxt.Image column",
                "Insert images from directory or URLs",
                "Add YOLOX for object detection: 'yolox.yolox(image, threshold=0.5)'",
                "Add OpenAI Vision: 'openai.vision(prompt, image)'",
                "Query results with filters"
            ],
            "3_rag_pipeline": [
                "Create table with pxt.Document column",
                "Insert documents (PDFs, text files)",
                "Add embedding column: 'sentence_transformers.embed(text)'",
                "Create embedding index for vector search",
                "Query with similarity search"
            ],
            "4_video_analysis": [
                "Create table with pxt.Video column",
                "Extract frames as computed column",
                "Run models on frames (object detection, classification)",
                "Aggregate results across frames"
            ]
        },
        
        "ai_integrations": {
            "cloud_providers": ["OpenAI (GPT, DALL-E, Whisper)", "Anthropic (Claude)", "Google (Gemini)", "Fireworks"],
            "local_models": ["Ollama", "Sentence Transformers", "YOLOX", "Hugging Face models"],
            "custom_models": "Create UDFs with pixeltable_create_udf() for any Python function"
        },
        
        "best_practices": [
            "Use computed columns instead of manual processing - they auto-update",
            "Leverage incremental computation - only new data gets processed",
            "Create views for different use cases instead of duplicating data",
            "Use snapshots before major changes for rollback capability",
            "Set appropriate num_retained_versions to manage storage"
        ],
        
        "common_patterns": {
            "batch_inference": "Add computed column → Pixeltable handles batching automatically",
            "model_comparison": "Add multiple computed columns with different models → Query to compare",
            "data_validation": "Use computed columns with Python expressions for validation rules",
            "feature_engineering": "Chain computed columns for complex transformations"
        },
        
        "getting_started": {
            "simple_example": {
                "description": "Analyze images with AI",
                "steps": [
                    "pixeltable_init() - Initialize Pixeltable",
                    "pixeltable_create_table('images', {'image': 'Image', 'label': 'String'})",
                    "pixeltable_insert_data('images', [{'image': 'cat.jpg', 'label': 'cat'}])",
                    "pixeltable_add_computed_column('images', 'objects', 'yolox.yolox(image)')",
                    "pixeltable_query_table('images') - See detected objects"
                ]
            }
        },
        
        "tips": [
            "Check dependencies before adding AI columns: pixeltable_check_dependencies()",
            "Use pixeltable_list_tools() to see all available operations",
            "Set custom datastore path: pixeltable_set_datastore('/my/path')",
            "Use execute_python() for interactive exploration with pxt pre-loaded"
        ]
    }

def pixeltable_list_tools() -> Dict[str, Any]:
    """
    List all available Pixeltable MCP tools with their descriptions.
    
    Returns a categorized list of all tools available in this MCP server,
    including their names and descriptions extracted from docstrings.
    
    Returns:
        Dict containing categorized tools and their descriptions
    """
    try:
        # Import all function modules to get access to the functions
        import inspect
        import mcp_server_pixeltable_stio.core.pixeltable_functions as pf
        import mcp_server_pixeltable_stio.core.repl_functions as rf
        
        # Get all pixeltable functions
        pixeltable_funcs = [
            (name, func) for name, func in inspect.getmembers(pf)
            if name.startswith('pixeltable_') and callable(func)
        ]
        
        # Get all REPL functions  
        repl_funcs = [
            (name, func) for name, func in inspect.getmembers(rf)
            if callable(func) and not name.startswith('_')
        ]
        
        # Define categories for better organization
        categories = {
            "Table Management": ["create_table", "get_table", "list_tables", "drop_table", "query_table", "get_table_schema"],
            "Data Operations": ["insert_data", "add_computed_column", "create_view", "create_snapshot", "create_replica"],
            "Directory Management": ["create_dir", "drop_dir", "list_dirs", "ls", "move"],
            "Configuration": ["init", "set_datastore", "get_datastore", "configure_logging", "get_version", "list_tools"],
            "AI/ML Integration": ["create_udf", "create_array", "create_tools", "connect_mcp", "query"],
            "Data Types": ["create_image_type", "create_video_type", "create_audio_type", "create_array_type", "create_json_type"],
            "Dependencies": ["check_dependencies", "install_yolox", "install_openai", "install_huggingface", "install_all_dependencies", "smart_install"],
            "REPL & Debug": ["execute_python", "introspect_function", "list_available_functions", "log_bug", "log_missing_feature", "generate_bug_report"],
            "Utilities": ["list_functions", "get_types", "system_diagnostics", "auto_install_for_expression", "suggest_install_from_error"],
        }
        
        # Initialize categorized tools
        categorized = {cat: [] for cat in categories}
        categorized["Other"] = []
        
        # Process all functions
        all_funcs = pixeltable_funcs + repl_funcs
        
        for name, func in all_funcs:
            # Get description from docstring
            doc = ""
            if func.__doc__:
                # Get first non-empty line of docstring
                lines = func.__doc__.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        doc = line
                        break
            
            # Categorize the function
            found = False
            for cat, keywords in categories.items():
                if any(kw in name.lower() for kw in keywords):
                    categorized[cat].append({
                        "name": name,
                        "description": doc or "No description available"
                    })
                    found = True
                    break
            
            if not found:
                categorized["Other"].append({
                    "name": name,
                    "description": doc or "No description available"
                })
        
        # Remove empty categories and sort tools within each category
        result = {}
        total_count = 0
        for cat, tools in categorized.items():
            if tools:
                result[cat] = sorted(tools, key=lambda x: x["name"])
                total_count += len(tools)
        
        return {
            "success": True,
            "total_tools": total_count,
            "categories": result,
            "message": f"Found {total_count} tools across {len(result)} categories"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def pixeltable_search_docs(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search Pixeltable documentation using Mintlify's MCP endpoint.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dict with search results including titles, links, and snippets
    """
    try:
        import requests
        import json

        # Call Mintlify MCP endpoint
        response = requests.post(
            "https://docs.pixeltable.com/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json"
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "SearchPixeltableDocumentation",
                    "arguments": {"query": query}
                }
            },
            timeout=10
        )

        # Parse SSE response
        results = []
        for line in response.text.split('\n'):
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'result' in data and 'content' in data['result']:
                    for item in data['result']['content'][:max_results]:
                        text = item.get('text', '')
                        lines = text.split('\n')

                        # Parse title and link
                        title = ''
                        link = ''
                        content = ''

                        for i, line_text in enumerate(lines):
                            if line_text.startswith('Title: '):
                                title = line_text[7:]
                            elif line_text.startswith('Link: '):
                                link = line_text[6:]
                            elif line_text.startswith('Content: '):
                                # Everything after "Content: " is the actual content
                                content = '\n'.join(lines[i:]).replace('Content: ', '', 1)
                                break

                        if title and link:
                            results.append({
                                'title': title,
                                'link': link,
                                'snippet': content[:500] if content else 'No content available'
                            })
                    break

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results,
            "message": f"Found {len(results)} results for '{query}'"
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Network error searching docs: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error searching docs: {str(e)}"
        }
