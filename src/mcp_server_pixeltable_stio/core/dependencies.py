"""
Dependency management for the Pixeltable MCP server.

Handles checking, installing, and auto-resolving Python package
dependencies required by Pixeltable AI/ML integrations.
"""

import logging
import platform
import subprocess
import sys
from typing import Any, Dict

from .helpers import pxt, ensure_pixeltable_available

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_uv_available() -> bool:
    """Check if uv is available in the system."""
    try:
        result = subprocess.run(
            ['uv', '--version'], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _run_uv_install(package: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run ``uv pip install`` with proper error handling."""
    if not _check_uv_available():
        raise RuntimeError("uv is not available in the system")

    python_path = sys.executable
    return subprocess.run(
        ['uv', 'pip', 'install', '--python', python_path, package],
        capture_output=True, text=True, timeout=timeout,
    )


def _direct_uv_install(package: str) -> Dict[str, Any]:
    """Install a package directly with uv – helper for common packages."""
    try:
        logger.info(f"Installing {package} directly with uv...")
        result = _run_uv_install(package, timeout=300)

        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Failed to install {package} with uv: {result.stderr}',
                'stdout': result.stdout,
            }

        return {
            'success': True,
            'message': f'Successfully installed {package} with uv',
            'package': package,
            'method': 'direct_uv',
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Installation of {package} timed out after 5 minutes',
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Installation of {package} failed: {e}',
        }


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------

def check_dependencies(expression: str) -> Dict[str, Any]:
    """Check what dependencies are needed for an expression."""
    missing = []
    available = []

    expr_lower = expression.lower()

    # YOLOX
    if 'yolox' in expr_lower:
        try:
            from pixeltable.ext.functions import yolox  # noqa: F401
            available.append('yolox')
        except ImportError:
            missing.append({
                'name': 'yolox',
                'packages': ['torch', 'torchvision', 'pixeltable-yolox'],
                'size': '~2.5GB',
                'time': '5-10 minutes',
                'description': 'YOLO object detection',
            })

    # OpenAI
    if any(term in expr_lower for term in ['openai', 'gpt']):
        try:
            from pixeltable.functions import openai  # noqa: F401
            available.append('openai')
        except ImportError:
            missing.append({
                'name': 'openai',
                'packages': ['openai'],
                'size': '~50MB',
                'time': '1-2 minutes',
                'description': 'OpenAI API integration',
            })

    # Hugging Face
    if any(term in expr_lower for term in ['huggingface', 'transformers']):
        try:
            from pixeltable.functions import huggingface  # noqa: F401
            available.append('huggingface')
        except ImportError:
            missing.append({
                'name': 'huggingface',
                'packages': ['transformers', 'torch'],
                'size': '~1.5GB',
                'time': '3-5 minutes',
                'description': 'Hugging Face transformers',
            })

    # Anthropic
    if any(term in expr_lower for term in ['anthropic', 'claude']):
        try:
            from pixeltable.functions import anthropic  # noqa: F401
            available.append('anthropic')
        except ImportError:
            missing.append({
                'name': 'anthropic',
                'packages': ['anthropic'],
                'size': '~20MB',
                'time': '1 minute',
                'description': 'Anthropic Claude API integration',
            })

    # Other AI services
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
                __import__(f'pixeltable.functions.{service}', fromlist=[service])
                available.append(service)
            except ImportError:
                missing.append({
                    'name': service,
                    'packages': info['packages'],
                    'size': '~20MB',
                    'time': '1 minute',
                    'description': info['description'],
                })

    # Audio / speech
    if any(term in expr_lower for term in ['whisper', 'speech', 'audio']):
        try:
            from pixeltable.ext.functions import whisperx  # noqa: F401
            available.append('whisperx')
        except ImportError:
            missing.append({
                'name': 'whisper',
                'packages': ['openai-whisper'],
                'size': '~100MB',
                'time': '2-3 minutes',
                'description': 'OpenAI Whisper speech recognition',
            })

    return {
        'missing': missing,
        'available': available,
        'all_satisfied': len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# Individual installers (used internally by smart_install)
# ---------------------------------------------------------------------------

def _install_yolox() -> Dict[str, Any]:
    """Install YOLOX dependencies for object detection using uv."""
    try:
        logger.info("Installing YOLOX dependencies with uv...")
        packages = ['torch', 'torchvision', 'pixeltable-yolox']

        for package in packages:
            logger.info(f"Installing {package} with uv...")
            result = _run_uv_install(package, timeout=600)
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'Failed to install {package} with uv: {result.stderr}',
                    'stdout': result.stdout,
                }
            logger.info(f"Successfully installed {package}")

        try:
            from pixeltable.ext.functions import yolox  # noqa: F401
            return {
                'success': True,
                'message': 'YOLOX installed successfully with uv! Object detection is now available.',
                'installed_packages': packages,
                'method': 'uv',
            }
        except ImportError as e:
            return {
                'success': False,
                'error': f'Installation completed but import failed: {e}',
            }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Installation timed out after 10 minutes'}
    except Exception as e:
        return {'success': False, 'error': f'Installation failed: {e}'}


def _install_openai() -> Dict[str, Any]:
    """Install OpenAI dependencies using uv."""
    try:
        logger.info("Installing OpenAI dependencies with uv...")
        result = _run_uv_install('openai', timeout=300)

        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Failed to install openai with uv: {result.stderr}',
                'stdout': result.stdout,
            }
        logger.info("Successfully installed openai")

        try:
            from pixeltable.functions import openai  # noqa: F401
            return {
                'success': True,
                'message': 'OpenAI installed successfully with uv! Vision and chat functions are now available.',
                'installed_packages': ['openai'],
                'method': 'uv',
            }
        except ImportError as e:
            return {
                'success': False,
                'error': f'Installation completed but import failed: {e}',
            }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Installation timed out after 5 minutes'}
    except Exception as e:
        return {'success': False, 'error': f'Installation failed: {e}'}


def _install_huggingface() -> Dict[str, Any]:
    """Install Hugging Face dependencies using uv."""
    try:
        logger.info("Installing Hugging Face dependencies with uv...")
        packages = ['transformers', 'torch']

        for package in packages:
            logger.info(f"Installing {package} with uv...")
            result = _run_uv_install(package, timeout=600)
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'Failed to install {package} with uv: {result.stderr}',
                    'stdout': result.stdout,
                }
            logger.info(f"Successfully installed {package}")

        try:
            from pixeltable.functions import huggingface  # noqa: F401
            return {
                'success': True,
                'message': 'Hugging Face installed successfully with uv! Transformers and models are now available.',
                'installed_packages': packages,
                'method': 'uv',
            }
        except ImportError as e:
            return {
                'success': False,
                'error': f'Installation completed but import failed: {e}',
            }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Installation timed out after 10 minutes'}
    except Exception as e:
        return {'success': False, 'error': f'Installation failed: {e}'}


def _install_all_dependencies() -> Dict[str, Any]:
    """Install all available AI/ML dependencies using uv."""
    try:
        logger.info("Starting bulk installation of all AI/ML dependencies with uv...")
        results = {}

        results['yolox'] = _install_yolox()
        results['openai'] = _install_openai()
        results['huggingface'] = _install_huggingface()

        success_count = sum(1 for r in results.values() if r.get('success', False))
        total_count = len(results)

        all_packages = []
        for r in results.values():
            if r.get('success') and 'installed_packages' in r:
                all_packages.extend(r['installed_packages'])

        return {
            'success': success_count > 0,
            'message': f'Installed {success_count}/{total_count} dependency groups with uv',
            'method': 'uv',
            'results': results,
            'fully_successful': success_count == total_count,
            'total_packages': list(set(all_packages)),
            'package_count': len(set(all_packages)),
        }

    except Exception as e:
        return {'success': False, 'error': f'Bulk installation failed: {e}'}


# ---------------------------------------------------------------------------
# Smart / auto installers
# ---------------------------------------------------------------------------

def _smart_install(package_hint: str) -> Dict[str, Any]:
    """Smart dependency installer that maps common package names to install routines."""
    try:
        hint = package_hint.lower().strip()

        install_map = {
            # YOLOX / object detection
            'yolox': _install_yolox,
            'yolo': _install_yolox,
            'object_detection': _install_yolox,
            'detection': _install_yolox,
            'torch': _install_yolox,
            'pytorch': _install_yolox,
            'torchvision': _install_yolox,
            # OpenAI
            'openai': _install_openai,
            'gpt': _install_openai,
            'vision': _install_openai,
            'chat': _install_openai,
            # Hugging Face
            'huggingface': _install_huggingface,
            'transformers': _install_huggingface,
            'hf': _install_huggingface,
            'bert': _install_huggingface,
            'llm': _install_huggingface,
            'sentence-transformers': lambda: _direct_uv_install('sentence-transformers'),
            'sentence_transformers': lambda: _direct_uv_install('sentence-transformers'),
            # AI API services
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
            # Audio / speech
            'whisper': lambda: _direct_uv_install('openai-whisper'),
            'openai-whisper': lambda: _direct_uv_install('openai-whisper'),
            'whisperx': lambda: _direct_uv_install('whisperx'),
            'speech': lambda: _direct_uv_install('openai-whisper'),
            'audio': lambda: _direct_uv_install('openai-whisper'),
            # Common packages
            'ollama': lambda: _direct_uv_install('ollama'),
            'pillow': lambda: _direct_uv_install('pillow'),
            'numpy': lambda: _direct_uv_install('numpy'),
            'pandas': lambda: _direct_uv_install('pandas'),
        }

        if hint in install_map:
            logger.info(f"Smart installing {hint} using mapped installer")
            return install_map[hint]()

        # Fallback: try raw uv install
        logger.info(f"No mapping found for '{hint}', attempting direct uv install")

        if not _check_uv_available():
            return {
                'success': False,
                'error': 'uv is not available and no mapping found for this package',
            }

        result = _run_uv_install(hint, timeout=300)

        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Failed to install {hint} with uv: {result.stderr}',
                'stdout': result.stdout,
                'suggestion': 'Try one of the supported packages: yolox, openai, huggingface',
            }

        return {
            'success': True,
            'message': f'Successfully installed {hint} with uv',
            'method': 'direct_uv',
            'package': hint,
        }

    except Exception as e:
        return {'success': False, 'error': f'Smart install failed: {e}'}


def pixeltable_auto_install_for_expression(expression: str) -> Dict[str, Any]:
    """Automatically detect and install dependencies needed for an expression."""
    try:
        deps = check_dependencies(expression)

        if deps['all_satisfied']:
            return {
                'success': True,
                'message': 'All dependencies already satisfied',
                'available': deps['available'],
                'installed': [],
            }

        logger.info(f"Auto-installing {len(deps['missing'])} missing dependencies for expression")

        installed = []
        failed = []

        for dep in deps['missing']:
            logger.info(f"Installing {dep['name']}...")
            result = _smart_install(dep['name'])

            if result.get('success', False):
                installed.append(dep['name'])
                logger.info(f"Successfully installed {dep['name']}")
            else:
                failed.append({
                    'name': dep['name'],
                    'error': result.get('error', 'Unknown error'),
                })
                logger.error(f"Failed to install {dep['name']}: {result.get('error')}")

        if failed:
            return {
                'success': len(installed) > 0,
                'message': f'Installed {len(installed)}/{len(deps["missing"])} dependencies',
                'installed': installed,
                'failed': failed,
                'partial': True,
            }

        return {
            'success': True,
            'message': f'Successfully installed all {len(installed)} missing dependencies',
            'installed': installed,
        }

    except Exception as e:
        return {'success': False, 'error': f'Auto-install failed: {e}'}


def _suggest_install_from_error(error_message: str) -> Dict[str, Any]:
    """Analyze an error message and suggest installations."""
    import re

    try:
        error_lower = error_message.lower()
        suggestions = []

        if 'no module named' in error_lower:
            if any(term in error_lower for term in ['yolox', 'torch', 'torchvision']):
                suggestions.append({
                    'package': 'yolox',
                    'function': 'pixeltable_install_dependency',
                    'description': 'Install YOLOX for object detection',
                })
            if 'openai' in error_lower:
                suggestions.append({
                    'package': 'openai',
                    'function': 'pixeltable_install_dependency',
                    'description': 'Install OpenAI for vision and chat functions',
                })
            if any(term in error_lower for term in ['transformers', 'huggingface']):
                suggestions.append({
                    'package': 'huggingface',
                    'function': 'pixeltable_install_dependency',
                    'description': 'Install Hugging Face transformers',
                })

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
                        'function': 'pixeltable_install_dependency',
                        'description': description,
                    })

            if any(term in error_lower for term in ['whisper', 'whisperx']):
                suggestions.append({
                    'package': 'whisper',
                    'function': 'pixeltable_install_dependency',
                    'description': 'Install Whisper for speech recognition',
                })

            match = re.search(r"no module named ['\"]([^'\"]+)['\"]?", error_lower)
            if match:
                package_name = match.group(1)
                suggestions.append({
                    'package': package_name,
                    'function': 'pixeltable_install_dependency',
                    'description': f'Try smart install for {package_name}',
                })

        if suggestions:
            return {
                'success': True,
                'suggestions': suggestions,
                'message': f'Found {len(suggestions)} installation suggestion(s)',
            }

        return {
            'success': True,
            'suggestions': [],
            'message': 'No installation suggestions found for this error',
        }

    except Exception as e:
        return {'success': False, 'error': f'Failed to analyze error: {e}'}


# ---------------------------------------------------------------------------
# PUBLIC TOOLS (registered as MCP tools)
# ---------------------------------------------------------------------------

def pixeltable_check_dependencies(expression: str) -> Dict[str, Any]:
    """Check what dependencies are needed for a Pixeltable expression.

    Inspects an expression string and reports which AI/ML packages are
    available and which are missing.

    Args:
        expression: A Pixeltable expression string to analyse
            (e.g. "yolox.yolox(table.image)")

    Returns:
        Dict with available/missing dependency information.
    """
    try:
        deps = check_dependencies(expression)

        if deps['all_satisfied']:
            return {
                'success': True,
                'message': 'All dependencies satisfied',
                'available': deps['available'],
            }

        return {
            'success': True,
            'dependencies_needed': True,
            'missing': deps['missing'],
            'message': f"Need to install {len(deps['missing'])} dependency group(s)",
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def pixeltable_install_dependency(name_or_expression: str) -> Dict[str, Any]:
    """Install a dependency by name, hint, or expression analysis.

    This is the single entry-point for all dependency management.  It handles:
    - Named packages: ``pixeltable_install_dependency("openai")``
    - Hint aliases: ``pixeltable_install_dependency("yolox")``
    - Expression analysis: ``pixeltable_install_dependency("openai.chat_completions(...)")``
    - Bulk install: ``pixeltable_install_dependency("all")``
    - Error-based: ``pixeltable_install_dependency("No module named 'torch'")``

    Args:
        name_or_expression: A package name, hint, Pixeltable expression,
            error message, or "all" to install every known dependency group.

    Returns:
        Dict with success status, installed packages, and details.
    """
    try:
        text = name_or_expression.strip()
        text_lower = text.lower()

        # "all" shortcut
        if text_lower == 'all':
            return _install_all_dependencies()

        # Error-message detection
        if 'no module named' in text_lower or 'importerror' in text_lower:
            suggestions = _suggest_install_from_error(text)
            if suggestions.get('suggestions'):
                installed = []
                failed = []
                for sug in suggestions['suggestions']:
                    pkg = sug['package']
                    result = _smart_install(pkg)
                    if result.get('success'):
                        installed.append(pkg)
                    else:
                        failed.append({'name': pkg, 'error': result.get('error', '')})
                return {
                    'success': len(installed) > 0,
                    'message': f'Installed {len(installed)} package(s) from error analysis',
                    'installed': installed,
                    'failed': failed if failed else None,
                }

        # Expression-based detection
        if '(' in text or (
            '.' in text
            and any(
                kw in text_lower
                for kw in ['table.', 'openai.', 'yolox.', 'huggingface.', 'whisper.']
            )
        ):
            return pixeltable_auto_install_for_expression(text)

        # Direct name / hint install
        return _smart_install(text)

    except Exception as e:
        return {'success': False, 'error': f'install_dependency failed: {e}'}


# ---------------------------------------------------------------------------
# System diagnostics (uses _check_uv_available, so lives here)
# ---------------------------------------------------------------------------

def pixeltable_system_diagnostics() -> Dict[str, Any]:
    """Get system diagnostics for Pixeltable and its dependencies."""
    try:
        diagnostics: Dict[str, Any] = {
            'pixeltable_version': None,
            'uv_available': False,
            'uv_version': None,
            'dependencies': {
                'yolox': False,
                'openai': False,
                'huggingface': False,
            },
            'system_info': {},
        }

        try:
            import pixeltable as _pxt
            diagnostics['pixeltable_version'] = _pxt.__version__
        except ImportError:
            diagnostics['pixeltable_version'] = 'Not installed'

        if _check_uv_available():
            diagnostics['uv_available'] = True
            try:
                result = subprocess.run(
                    ['uv', '--version'], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    diagnostics['uv_version'] = result.stdout.strip()
            except Exception:
                pass

        try:
            from pixeltable.ext.functions import yolox  # noqa: F401
            diagnostics['dependencies']['yolox'] = True
        except ImportError:
            pass

        try:
            from pixeltable.functions import openai  # noqa: F401
            diagnostics['dependencies']['openai'] = True
        except ImportError:
            pass

        try:
            from pixeltable.functions import huggingface  # noqa: F401
            diagnostics['dependencies']['huggingface'] = True
        except ImportError:
            pass

        diagnostics['system_info'] = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
        }

        return {'success': True, 'diagnostics': diagnostics}

    except Exception as e:
        return {'success': False, 'error': str(e)}
