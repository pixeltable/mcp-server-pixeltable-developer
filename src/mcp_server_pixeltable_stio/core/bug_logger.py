"""
Bug logging functionality for PixelTable MCP server.

Provides structured logging of issues and discoveries during testing.
"""

import os
import json
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BugLogger:
    """Manages structured logging of bugs and issues."""
    
    def __init__(self, log_directory: Optional[str] = None):
        """
        Initialize bug logger.
        
        Args:
            log_directory: Directory to store logs. If None, uses project directory.
        """
        if log_directory is None:
            # Default to project root directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            log_directory = project_root / "pixeltable_testing_logs"
        
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Session log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_directory / f"session_{timestamp}.md"
        self.bug_log_file = self.log_directory / "pixeltable_bugs.md"
        self.json_log_file = self.log_directory / "pixeltable_bugs.json"
        
        # Initialize session log
        self._initialize_session_log()
        
        logger.info(f"Bug logger initialized with directory: {self.log_directory}")
    
    def _initialize_session_log(self):
        """Initialize the session log file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""# PixelTable Testing Session - {timestamp}

## Session Information
- **Start Time**: {timestamp}
- **Log Directory**: {self.log_directory}
- **Bug Log File**: {self.bug_log_file}

## Session Log

"""
        with open(self.session_log_file, 'w') as f:
            f.write(header)
    
    def log_bug(self, 
                description: str, 
                severity: str = "medium",
                category: str = "general",
                attempted_action: str = "",
                expected_result: str = "",
                actual_result: str = "",
                workaround: str = "",
                function_name: str = "",
                error_message: str = "") -> Dict[str, Any]:
        """
        Log a bug or issue.
        
        Args:
            description: Description of the bug
            severity: Bug severity (low, medium, high, critical)
            category: Bug category (api, documentation, performance, etc.)
            attempted_action: What was attempted
            expected_result: What was expected to happen
            actual_result: What actually happened
            workaround: Any workaround found
            function_name: Name of the function that had issues
            error_message: Any error message received
            
        Returns:
            Dictionary with bug entry details
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        bug_entry = {
            "timestamp": timestamp,
            "description": description,
            "severity": severity,
            "category": category,
            "attempted_action": attempted_action,
            "expected_result": expected_result,
            "actual_result": actual_result,
            "workaround": workaround,
            "function_name": function_name,
            "error_message": error_message,
            "session_file": str(self.session_log_file.name)
        }
        
        # Write to session log
        self._write_to_session_log(bug_entry)
        
        # Write to main bug log
        self._write_to_bug_log(bug_entry)
        
        # Write to JSON log
        self._write_to_json_log(bug_entry)
        
        logger.info(f"Logged bug: {description[:50]}...")
        
        return bug_entry
    
    def _write_to_session_log(self, bug_entry: Dict[str, Any]):
        """Write bug entry to session log."""
        markdown_entry = f"""
### Bug Report - {bug_entry['timestamp']}

**Severity**: {bug_entry['severity']} | **Category**: {bug_entry['category']}

**Description**: {bug_entry['description']}

"""
        if bug_entry['function_name']:
            markdown_entry += f"**Function**: `{bug_entry['function_name']}`\n\n"
        
        if bug_entry['attempted_action']:
            markdown_entry += f"**Attempted Action**: {bug_entry['attempted_action']}\n\n"
        
        if bug_entry['expected_result']:
            markdown_entry += f"**Expected Result**: {bug_entry['expected_result']}\n\n"
        
        if bug_entry['actual_result']:
            markdown_entry += f"**Actual Result**: {bug_entry['actual_result']}\n\n"
        
        if bug_entry['error_message']:
            markdown_entry += f"**Error Message**:\n```\n{bug_entry['error_message']}\n```\n\n"
        
        if bug_entry['workaround']:
            markdown_entry += f"**Workaround**: {bug_entry['workaround']}\n\n"
        
        markdown_entry += "---\n"
        
        with open(self.session_log_file, 'a') as f:
            f.write(markdown_entry)
    
    def _write_to_bug_log(self, bug_entry: Dict[str, Any]):
        """Write bug entry to main bug log."""
        # Check if main bug log exists, create header if not
        if not self.bug_log_file.exists():
            header = """# PixelTable Bug Log

This file contains all reported bugs and issues discovered during testing.

## Bug Reports

"""
            with open(self.bug_log_file, 'w') as f:
                f.write(header)
        
        # Write the bug entry
        self._write_to_session_log(bug_entry)  # Same format
        
        # Also append to main log
        with open(self.bug_log_file, 'a') as f:
            f.write(f"\n**Session**: {bug_entry['session_file']}\n")
            f.write("---\n")
    
    def _write_to_json_log(self, bug_entry: Dict[str, Any]):
        """Write bug entry to JSON log for programmatic access."""
        # Load existing entries
        entries = []
        if self.json_log_file.exists():
            try:
                with open(self.json_log_file, 'r') as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                entries = []
        
        # Add new entry
        entries.append(bug_entry)
        
        # Write back to file
        with open(self.json_log_file, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def log_missing_feature(self, 
                          feature_description: str,
                          use_case: str = "",
                          attempted_approach: str = "",
                          api_expectation: str = "") -> Dict[str, Any]:
        """
        Log a missing feature or functionality.
        
        Args:
            feature_description: Description of the missing feature
            use_case: What you were trying to accomplish
            attempted_approach: How you tried to achieve it
            api_expectation: What API you expected to exist
            
        Returns:
            Dictionary with feature request details
        """
        return self.log_bug(
            description=f"Missing Feature: {feature_description}",
            severity="medium",
            category="missing_feature",
            attempted_action=attempted_approach,
            expected_result=f"Expected API: {api_expectation}" if api_expectation else "",
            actual_result="Feature not available",
            function_name="N/A",
            error_message=f"Use case: {use_case}"
        )
    
    def log_success(self, 
                   description: str,
                   function_name: str = "",
                   approach: str = "",
                   notes: str = "") -> Dict[str, Any]:
        """
        Log a successful operation or discovery.
        
        Args:
            description: Description of what worked
            function_name: Function that worked
            approach: Approach that worked
            notes: Additional notes
            
        Returns:
            Dictionary with success entry details
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        success_entry = {
            "timestamp": timestamp,
            "type": "success",
            "description": description,
            "function_name": function_name,
            "approach": approach,
            "notes": notes
        }
        
        # Write to session log
        markdown_entry = f"""
### ✅ Success - {timestamp}

**Description**: {description}

"""
        if function_name:
            markdown_entry += f"**Function**: `{function_name}`\n\n"
        
        if approach:
            markdown_entry += f"**Approach**: {approach}\n\n"
        
        if notes:
            markdown_entry += f"**Notes**: {notes}\n\n"
        
        markdown_entry += "---\n"
        
        with open(self.session_log_file, 'a') as f:
            f.write(markdown_entry)
        
        logger.info(f"Logged success: {description[:50]}...")
        
        return success_entry
    
    def generate_bug_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of all logged bugs.
        
        Returns:
            Dictionary with bug report summary
        """
        if not self.json_log_file.exists():
            return {"total_bugs": 0, "bugs_by_severity": {}, "bugs_by_category": {}}
        
        try:
            with open(self.json_log_file, 'r') as f:
                entries = json.load(f)
            
            # Analyze bugs
            total_bugs = len(entries)
            bugs_by_severity = {}
            bugs_by_category = {}
            bugs_by_function = {}
            
            for entry in entries:
                severity = entry.get('severity', 'unknown')
                category = entry.get('category', 'unknown')
                function_name = entry.get('function_name', 'unknown')
                
                bugs_by_severity[severity] = bugs_by_severity.get(severity, 0) + 1
                bugs_by_category[category] = bugs_by_category.get(category, 0) + 1
                bugs_by_function[function_name] = bugs_by_function.get(function_name, 0) + 1
            
            return {
                "total_bugs": total_bugs,
                "bugs_by_severity": bugs_by_severity,
                "bugs_by_category": bugs_by_category,
                "bugs_by_function": bugs_by_function,
                "latest_bugs": entries[-5:] if entries else []  # Last 5 bugs
            }
        
        except Exception as e:
            logger.error(f"Error generating bug report: {e}")
            return {"error": str(e)}
    
    def get_session_summary(self) -> str:
        """Get a summary of the current session."""
        if not self.session_log_file.exists():
            return "No session log found"
        
        try:
            with open(self.session_log_file, 'r') as f:
                content = f.read()
            
            # Count entries
            bug_count = content.count("### Bug Report")
            success_count = content.count("### ✅ Success")
            
            return f"""Session Summary:
- Session log: {self.session_log_file.name}
- Bugs reported: {bug_count}
- Successes logged: {success_count}
- Log file size: {len(content)} characters
"""
        except Exception as e:
            return f"Error reading session summary: {e}"


# Global bug logger instance
_bug_logger_instance = None


def get_bug_logger() -> BugLogger:
    """Get or create the global bug logger instance."""
    global _bug_logger_instance
    if _bug_logger_instance is None:
        try:
            _bug_logger_instance = BugLogger()
        except Exception as e:
            logger.error(f"Failed to initialize bug logger: {e}")
            # Return a dummy logger that doesn't actually log
            class DummyBugLogger:
                def log_bug(self, *args, **kwargs):
                    return {"error": "Bug logger not initialized"}
                def log_missing_feature(self, *args, **kwargs):
                    return {"error": "Bug logger not initialized"}
                def log_success(self, *args, **kwargs):
                    return {"error": "Bug logger not initialized"}
                def generate_bug_report(self, *args, **kwargs):
                    return {"error": "Bug logger not initialized"}
                def get_session_summary(self, *args, **kwargs):
                    return "Bug logger not initialized"
            _bug_logger_instance = DummyBugLogger()
    return _bug_logger_instance
