"""
REPL and bug logging functions for PixelTable MCP server.

These functions provide interactive Python execution and structured bug logging.
"""

from typing import Dict, Any, Optional
import logging

from mcp_server_pixeltable_stio.core.repl_session import get_repl_instance
from mcp_server_pixeltable_stio.core.bug_logger import get_bug_logger

logger = logging.getLogger(__name__)


def execute_python(code: str, reset: bool = False) -> Dict[str, Any]:
    """
    Execute Python code in a persistent REPL session with PixelTable pre-loaded.
    
    This function provides a persistent Python execution environment where:
    - PixelTable is pre-imported as 'pxt'
    - Variables persist between executions
    - Full Python standard library is available
    - Output is captured and returned
    
    Args:
        code: Python code to execute
        reset: If True, clear all session variables before execution
        
    Returns:
        Dictionary containing:
        - success: Whether execution was successful
        - output: Captured output from the execution
        - error: Error message if execution failed
        - stderr: Standard error output if any
        
    Examples:
        execute_python("import numpy as np; print(np.__version__)")
        execute_python("tables = pxt.list_tables(); print(f'Found {len(tables)} tables')")
        execute_python("x = 42", reset=True)  # Clear session first
    """
    try:
        repl = get_repl_instance()
        result = repl.execute_python(code, reset=reset)
        
        logger.info(f"Executed Python code: {code[:50]}{'...' if len(code) > 50 else ''}")
        
        return result
    except Exception as e:
        logger.error(f"Error in execute_python: {e}")
        return {
            "success": False,
            "error": str(e),
            "output": "",
            "stderr": ""
        }


def introspect_function(function_path: str) -> Dict[str, Any]:
    """
    Get documentation, signature, and details for any function or object.
    
    This function uses Python's inspect module to provide detailed information
    about functions, classes, or modules available in the REPL session.
    
    Args:
        function_path: Dot-separated path to the function/object (e.g., "pxt.create_table", "inspect.signature")
        
    Returns:
        Dictionary containing:
        - signature: Function signature as string
        - docstring: Function documentation
        - module: Module where function is defined
        - parameters: List of parameter names
        - error: Error message if introspection failed
        
    Examples:
        introspect_function("pxt.create_table")
        introspect_function("pxt.list_tables")
        introspect_function("inspect.getdoc")
    """
    try:
        repl = get_repl_instance()
        result = repl.introspect_function(function_path)
        
        logger.info(f"Introspected function: {function_path}")
        
        return result
    except Exception as e:
        logger.error(f"Error in introspect_function: {e}")
        return {
            "error": str(e),
            "signature": "Unknown",
            "docstring": "Could not retrieve documentation",
            "module": "Unknown",
            "parameters": []
        }


def list_available_functions(module_name: str = "pxt") -> Dict[str, Any]:
    """
    List all available functions in a specified module.
    
    This function helps discover what functions are available in a module,
    which is especially useful for exploring PixelTable's API.
    
    Args:
        module_name: Name of the module to inspect (default: "pxt" for PixelTable)
        
    Returns:
        Dictionary containing:
        - functions: List of available function names
        - error: Error message if listing failed
        
    Examples:
        list_available_functions("pxt")  # List PixelTable functions
        list_available_functions("os")   # List os module functions
    """
    try:
        repl = get_repl_instance()
        result = repl.list_available_functions(module_name)
        
        logger.info(f"Listed functions for module: {module_name}")
        
        return result
    except Exception as e:
        logger.error(f"Error in list_available_functions: {e}")
        return {
            "functions": [],
            "error": str(e)
        }


def install_package(package_name: str) -> Dict[str, Any]:
    """
    Install a Python package in the current environment.
    
    This function uses pip to install packages that can then be used
    in the REPL session.
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        Dictionary containing:
        - success: Whether installation was successful
        - output: Installation output
        - stderr: Error output if any
        - package: Name of the package that was installed
        
    Examples:
        install_package("numpy")
        install_package("pandas")
        install_package("matplotlib")
    """
    try:
        repl = get_repl_instance()
        result = repl.install_package(package_name)
        
        logger.info(f"Attempted to install package: {package_name}")
        
        return result
    except Exception as e:
        logger.error(f"Error in install_package: {e}")
        return {
            "success": False,
            "error": str(e),
            "package": package_name
        }


def log_bug(description: str,
           severity: str = "medium",
           category: str = "general",
           attempted_action: str = "",
           expected_result: str = "",
           actual_result: str = "",
           workaround: str = "",
           function_name: str = "",
           error_message: str = "") -> Dict[str, Any]:
    """
    Log a bug or issue encountered during testing.
    
    This function creates structured bug reports that are saved to both
    markdown and JSON formats for easy review and analysis.
    
    Args:
        description: Clear description of the bug
        severity: Bug severity ("low", "medium", "high", "critical")
        category: Bug category ("api", "documentation", "performance", "missing_feature", etc.)
        attempted_action: What you were trying to do
        expected_result: What you expected to happen
        actual_result: What actually happened
        workaround: Any workaround you found
        function_name: Name of the problematic function
        error_message: Any error message received
        
    Returns:
        Dictionary with the logged bug entry details
        
    Examples:
        log_bug("Cannot save images to table", 
                severity="high",
                function_name="pxt.create_table",
                attempted_action="Tried to create table with image column",
                expected_result="Table created with image column",
                actual_result="Got TypeError about image types")
    """
    try:
        bug_logger = get_bug_logger()
        result = bug_logger.log_bug(
            description=description,
            severity=severity,
            category=category,
            attempted_action=attempted_action,
            expected_result=expected_result,
            actual_result=actual_result,
            workaround=workaround,
            function_name=function_name,
            error_message=error_message
        )
        
        logger.info(f"Logged bug: {description[:50]}...")
        
        return result
    except Exception as e:
        logger.error(f"Error in log_bug: {e}")
        return {
            "error": str(e),
            "timestamp": "",
            "description": description
        }


def log_missing_feature(feature_description: str,
                       use_case: str = "",
                       attempted_approach: str = "",
                       api_expectation: str = "") -> Dict[str, Any]:
    """
    Log a missing feature or functionality in PixelTable.
    
    This function is specifically for documenting features that don't exist
    but would be useful, or APIs that you expected to find but couldn't.
    
    Args:
        feature_description: Description of the missing feature
        use_case: What you were trying to accomplish
        attempted_approach: How you tried to achieve it
        api_expectation: What API you expected to exist
        
    Returns:
        Dictionary with the logged feature request details
        
    Examples:
        log_missing_feature("No direct way to resize images in table",
                           use_case="Need to standardize image sizes before analysis",
                           attempted_approach="Looked for image.resize() function",
                           api_expectation="pxt.image.resize() or similar")
    """
    try:
        bug_logger = get_bug_logger()
        result = bug_logger.log_missing_feature(
            feature_description=feature_description,
            use_case=use_case,
            attempted_approach=attempted_approach,
            api_expectation=api_expectation
        )
        
        logger.info(f"Logged missing feature: {feature_description[:50]}...")
        
        return result
    except Exception as e:
        logger.error(f"Error in log_missing_feature: {e}")
        return {
            "error": str(e),
            "description": feature_description
        }


def log_success(description: str,
               function_name: str = "",
               approach: str = "",
               notes: str = "") -> Dict[str, Any]:
    """
    Log a successful operation or discovery.
    
    This function documents things that work well, successful approaches,
    and positive discoveries about PixelTable functionality.
    
    Args:
        description: Description of what worked
        function_name: Function that worked successfully
        approach: The approach that worked
        notes: Additional notes or observations
        
    Returns:
        Dictionary with the logged success entry details
        
    Examples:
        log_success("Successfully created table with mixed data types",
                   function_name="pxt.create_table",
                   approach="Used schema parameter with explicit types",
                   notes="Works well with images, text, and numeric data")
    """
    try:
        bug_logger = get_bug_logger()
        result = bug_logger.log_success(
            description=description,
            function_name=function_name,
            approach=approach,
            notes=notes
        )
        
        logger.info(f"Logged success: {description[:50]}...")
        
        return result
    except Exception as e:
        logger.error(f"Error in log_success: {e}")
        return {
            "error": str(e),
            "description": description
        }


def generate_bug_report() -> Dict[str, Any]:
    """
    Generate a summary report of all logged bugs and issues.
    
    This function analyzes all logged bugs and creates a summary report
    with statistics and categorization.
    
    Returns:
        Dictionary containing:
        - total_bugs: Total number of bugs logged
        - bugs_by_severity: Count of bugs by severity level
        - bugs_by_category: Count of bugs by category
        - bugs_by_function: Count of bugs by function
        - latest_bugs: Most recent 5 bug entries
        - error: Error message if report generation failed
        
    Examples:
        report = generate_bug_report()
        print(f"Total bugs: {report['total_bugs']}")
    """
    try:
        bug_logger = get_bug_logger()
        result = bug_logger.generate_bug_report()
        
        logger.info("Generated bug report")
        
        return result
    except Exception as e:
        logger.error(f"Error in generate_bug_report: {e}")
        return {
            "error": str(e),
            "total_bugs": 0
        }


def get_session_summary() -> Dict[str, Any]:
    """
    Get a summary of the current testing session.
    
    This function provides information about the current session,
    including number of bugs logged, successes recorded, and log file details.
    
    Returns:
        Dictionary containing:
        - summary: Text summary of the session
        - error: Error message if summary generation failed
        
    Examples:
        summary = get_session_summary()
        print(summary['summary'])
    """
    try:
        bug_logger = get_bug_logger()
        summary_text = bug_logger.get_session_summary()
        
        logger.info("Retrieved session summary")
        
        return {
            "summary": summary_text
        }
    except Exception as e:
        logger.error(f"Error in get_session_summary: {e}")
        return {
            "error": str(e),
            "summary": "Could not retrieve session summary"
        }
