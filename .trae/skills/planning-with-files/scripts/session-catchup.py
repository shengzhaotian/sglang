#!/usr/bin/env python3
"""
Session Catchup Script for planning-with-files (Trae Adapted Version)

Analyzes the previous session to find unsynced context after the last
planning file update. Designed to run on SessionStart.

Usage: python session-catchup.py [project-path]

Trae Adaptations:
- Uses TRAE_SESSION_PATH environment variable for session storage
- Falls back to standard Trae session locations
- Removed Claude Code specific path handling
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

PLANNING_FILES = ['task_plan.md', 'progress.md', 'findings.md']


def get_trae_session_dir(project_path: str) -> Optional[Path]:
    """
    Get Trae's session storage directory.
    Priority:
    1. TRAE_SESSION_PATH environment variable
    2. TRAE_SKILLS_PATH/../sessions (relative to skills path)
    3. Default: ~/.trae/sessions/
    """
    if 'TRAE_SESSION_PATH' in os.environ:
        return Path(os.environ['TRAE_SESSION_PATH'])
    
    if 'TRAE_SKILLS_PATH' in os.environ:
        skills_path = Path(os.environ['TRAE_SKILLS_PATH'])
        potential_session = skills_path.parent / 'sessions'
        if potential_session.exists():
            return potential_session
    
    default_path = Path.home() / '.trae' / 'sessions'
    if default_path.exists():
        return default_path
    
    return None


def get_project_session_dir(session_base: Path, project_path: str) -> Optional[Path]:
    """
    Get the session directory for a specific project.
    Trae may organize sessions differently than Claude Code.
    """
    project_name = Path(project_path).name
    
    potential_paths = [
        session_base / project_name,
        session_base / f"project-{project_name}",
        session_base,
    ]
    
    for path in potential_paths:
        if path.exists():
            return path
    
    return None


def get_sessions_sorted(session_dir: Path) -> List[Path]:
    """Get all session files sorted by modification time (newest first)."""
    if not session_dir:
        return []
    
    sessions = list(session_dir.glob('*.jsonl'))
    main_sessions = [s for s in sessions if not s.name.startswith('agent-')]
    return sorted(main_sessions, key=lambda p: p.stat().st_mtime, reverse=True)


def parse_session_messages(session_file: Path) -> List[Dict]:
    """Parse all messages from a session file, preserving order."""
    messages = []
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    data['_line_num'] = line_num
                    messages.append(data)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[planning-with-files] Warning: Could not read session file: {e}")
    return messages


def find_last_planning_update(messages: List[Dict]) -> Tuple[int, Optional[str]]:
    """
    Find the last time a planning file was written/edited.
    Returns (line_number, filename) or (-1, None) if not found.
    """
    last_update_line = -1
    last_update_file = None

    for msg in messages:
        msg_type = msg.get('type')

        if msg_type == 'assistant':
            content = msg.get('message', {}).get('content', [])
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'tool_use':
                        tool_name = item.get('name', '')
                        tool_input = item.get('input', {})

                        if tool_name in ('Write', 'Edit', 'SearchReplace'):
                            file_path = tool_input.get('file_path', '')
                            for pf in PLANNING_FILES:
                                if file_path.endswith(pf):
                                    last_update_line = msg['_line_num']
                                    last_update_file = pf

    return last_update_line, last_update_file


def extract_messages_after(messages: List[Dict], after_line: int) -> List[Dict]:
    """Extract conversation messages after a certain line number."""
    result = []
    for msg in messages:
        if msg['_line_num'] <= after_line:
            continue

        msg_type = msg.get('type')
        is_meta = msg.get('isMeta', False)

        if msg_type == 'user' and not is_meta:
            content = msg.get('message', {}).get('content', '')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content = item.get('text', '')
                        break
                else:
                    content = ''

            if content and isinstance(content, str):
                if content.startswith(('<local-command', '<command-', '<task-notification')):
                    continue
                if len(content) > 20:
                    result.append({'role': 'user', 'content': content, 'line': msg['_line_num']})

        elif msg_type == 'assistant':
            msg_content = msg.get('message', {}).get('content', '')
            text_content = ''
            tool_uses = []

            if isinstance(msg_content, str):
                text_content = msg_content
            elif isinstance(msg_content, list):
                for item in msg_content:
                    if item.get('type') == 'text':
                        text_content = item.get('text', '')
                    elif item.get('type') == 'tool_use':
                        tool_name = item.get('name', '')
                        tool_input = item.get('input', {})
                        if tool_name in ('Edit', 'SearchReplace'):
                            tool_uses.append(f"Edit: {tool_input.get('file_path', 'unknown')}")
                        elif tool_name == 'Write':
                            tool_uses.append(f"Write: {tool_input.get('file_path', 'unknown')}")
                        elif tool_name in ('Bash', 'RunCommand'):
                            cmd = tool_input.get('command', '')[:80]
                            tool_uses.append(f"Command: {cmd}")
                        else:
                            tool_uses.append(f"{tool_name}")

            if text_content or tool_uses:
                result.append({
                    'role': 'assistant',
                    'content': text_content[:600] if text_content else '',
                    'tools': tool_uses,
                    'line': msg['_line_num']
                })

    return result


def main():
    project_path = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    
    has_planning_files = any(
        Path(project_path, f).exists() for f in PLANNING_FILES
    )
    if not has_planning_files:
        return

    session_base = get_trae_session_dir(project_path)
    if not session_base:
        return

    session_dir = get_project_session_dir(session_base, project_path)
    if not session_dir:
        return

    sessions = get_sessions_sorted(session_dir)
    if len(sessions) < 1:
        return

    target_session = None
    for session in sessions:
        if session.stat().st_size > 5000:
            target_session = session
            break

    if not target_session:
        return

    messages = parse_session_messages(target_session)
    last_update_line, last_update_file = find_last_planning_update(messages)

    if last_update_line < 0:
        return

    messages_after = extract_messages_after(messages, last_update_line)

    if not messages_after:
        return

    print("\n[planning-with-files] SESSION CATCHUP DETECTED")
    print(f"Previous session: {target_session.stem}")

    print(f"Last planning update: {last_update_file} at message #{last_update_line}")
    print(f"Unsynced messages: {len(messages_after)}")

    print("\n--- UNSYNCED CONTEXT ---")
    for msg in messages_after[-15:]:
        if msg['role'] == 'user':
            print(f"USER: {msg['content'][:300]}")
        else:
            if msg.get('content'):
                print(f"ASSISTANT: {msg['content'][:300]}")
            if msg.get('tools'):
                print(f"  Tools: {', '.join(msg['tools'][:4])}")

    print("\n--- RECOMMENDED ---")
    print("1. Run: git diff --stat")
    print("2. Read: task_plan.md, progress.md, findings.md")
    print("3. Update planning files based on above context")
    print("4. Continue with task")


if __name__ == '__main__':
    main()
