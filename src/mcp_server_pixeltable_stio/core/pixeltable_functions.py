"""
Core Pixeltable functions for MCP server.

This module wraps the main Pixeltable API functions for use in the MCP server.
"""

import logging
import json
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

# Import pixeltable modules
try:
    import pixeltable as pxt
except ImportError as e:
    logging.error(f"Failed to import pixeltable: {e}")
    pxt = None

logger = logging.getLogger(__name__)

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
    
    return subprocess.run([
        'uv', 'pip', 'install', package
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
