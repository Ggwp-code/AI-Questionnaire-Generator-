#!/usr/bin/env python3
"""
Verify which agents/nodes actually exist in the graph_agent.py multi-agent pipeline.
Run this to see PROOF of all implemented agents.
"""

import re
from pathlib import Path

# Read the graph_agent.py file
graph_agent_file = Path("/home/user/AI-Questionnaire-Generator-/app/services/graph_agent.py")
content = graph_agent_file.read_text()

# Extract all workflow.add_node() calls
node_pattern = r'workflow\.add_node\("([^"]+)",\s*([^)]+)\)'
matches = re.findall(node_pattern, content)

print("=" * 70)
print("AI QUESTIONNAIRE GENERATOR - AGENT VERIFICATION")
print("=" * 70)
print()
print(f"Found {len(matches)} agents/nodes in graph_agent.py:")
print()

agents = []
for i, (node_name, function_name) in enumerate(matches, 1):
    # Find the function definition to get description
    func_pattern = rf'def {function_name.strip()}\([^)]*\):[^"]*"""([^"]*)'
    func_match = re.search(func_pattern, content)

    description = "No description found"
    if func_match:
        description = func_match.group(1).strip().split('\n')[0]

    agents.append({
        'number': i,
        'node_name': node_name,
        'function': function_name.strip(),
        'description': description
    })

    print(f"{i:2}. {node_name:20} → {function_name.strip():30}")

print()
print("=" * 70)
print("DETAILED AGENT DESCRIPTIONS:")
print("=" * 70)
print()

for agent in agents:
    print(f"Agent {agent['number']}: {agent['node_name'].upper()}")
    print(f"  Function: {agent['function']}")
    print(f"  Purpose: {agent['description']}")
    print()

print("=" * 70)
print("COMPARISON WITH PPT:")
print("=" * 70)
print()

ppt_agents = [
    ("Bloom Analyzer", "bloom_analyzer"),
    ("Scout", "scout"),
    ("Code Author", "code_author"),
    ("Theory Author", "theory_author"),
    ("Executor", "executor"),
    ("Question Author", "question_author"),
    ("Reviewer", "reviewer"),
    ("Pedagogy Tagger", "pedagogy_tagger"),
    ("Guardian", "guardian"),
    ("Archivist", "archivist"),
]

node_names = [m[0] for m in matches]

for ppt_name, expected_node in ppt_agents:
    status = "✅ EXISTS" if expected_node in node_names else "❌ MISSING"
    print(f"{status} - {ppt_name:20} (node: {expected_node})")

print()
print("=" * 70)
print("ADDITIONAL AGENTS (Not shown in PPT but exist in code):")
print("=" * 70)
print()

ppt_node_names = [n for _, n in ppt_agents]
extra_agents = [n for n in node_names if n not in ppt_node_names]

if extra_agents:
    for extra in extra_agents:
        print(f"  • {extra}")
else:
    print("  (None)")

print()
print("=" * 70)
print(f"TOTAL: {len(matches)} agents implemented in codebase")
print("=" * 70)
