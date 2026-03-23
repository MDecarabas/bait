"""Generate device_skills.md from a BITS instrument installation.

Parses device YAML configs and Python device classes (via AST) to produce
a structured markdown reference that AI agents can consume.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any

import autogen
import yaml

from tomobait.config import BaitConfig
from tomobait.utils import build_llm_config

# load_dotenv()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class DeviceInstance:
    """A device instantiated in devices.yml."""

    def __init__(
        self,
        class_path: str,
        name: str,
        prefix: str = "",
        labels: list[str] | None = None,
        kwargs: dict | None = None,
    ):
        self.class_path = class_path
        self.name = name
        self.prefix = prefix
        self.labels = labels or []
        self.kwargs = kwargs or {}


class ComponentInfo:
    """A single Component/FormattedComponent on a device class."""

    def __init__(
        self,
        attr_name: str,
        comp_type: str,
        signal_class: str,
        pv_suffix: str,
        kind: str = "",
        string: bool = False,
        is_subdevice: bool = False,
    ):
        self.attr_name = attr_name
        self.comp_type = comp_type  # "Cpt" or "FCpt"
        self.signal_class = signal_class
        self.pv_suffix = pv_suffix
        self.kind = kind
        self.string = string
        self.is_subdevice = is_subdevice


class DeviceClassInfo:
    """Parsed information about an ophyd Device class."""

    def __init__(
        self,
        name: str,
        module: str,
        docstring: str,
        bases: list[str],
        components: list[ComponentInfo],
    ):
        self.name = name
        self.module = module
        self.docstring = docstring
        self.bases = bases
        self.components = components


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def parse_devices_yaml(configs_dir: Path) -> list[DeviceInstance]:
    """Parse all YAML files in the configs directory for device instances."""
    instances: list[DeviceInstance] = []

    yaml_files = sorted(configs_dir.glob("devices*.yml"))
    if not yaml_files:
        yaml_files = sorted(configs_dir.glob("devices*.yaml"))

    for yaml_path in yaml_files:
        try:
            data = yaml.safe_load(yaml_path.read_text()) or {}
        except yaml.YAMLError as exc:
            print(f"  Warning: could not parse {yaml_path.name}: {exc}")
            continue

        for class_path, entries in data.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "")
                if not name:
                    continue
                instances.append(
                    DeviceInstance(
                        class_path=class_path,
                        name=name,
                        prefix=entry.get("prefix", ""),
                        labels=entry.get("labels", []),
                        kwargs={
                            k: v
                            for k, v in entry.items()
                            if k not in ("name", "prefix", "labels", "creator")
                        },
                    )
                )

    return instances


# ---------------------------------------------------------------------------
# AST parsing
# ---------------------------------------------------------------------------


def _resolve_constant(node: ast.expr) -> Any:
    """Extract a constant value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _parse_component_call(node: ast.Call) -> ComponentInfo | None:
    """Parse a Cpt(...) or FCpt(...) call into ComponentInfo."""
    # Determine component type
    func = node.func
    if isinstance(func, ast.Name):
        comp_type = func.id
    elif isinstance(func, ast.Attribute):
        comp_type = func.attr
    else:
        return None

    if comp_type not in ("Cpt", "FCpt", "Component", "FormattedComponent"):
        return None

    # Normalize
    if comp_type in ("Component", "Cpt"):
        comp_type = "Cpt"
    else:
        comp_type = "FCpt"

    if len(node.args) < 1:
        return None

    # First arg is the signal class (or sub-device class)
    signal_arg = node.args[0]
    if isinstance(signal_arg, ast.Name):
        signal_class = signal_arg.id
    elif isinstance(signal_arg, ast.Attribute):
        signal_class = signal_arg.attr
    else:
        signal_class = "Unknown"

    # Second positional arg is PV suffix (if present)
    pv_suffix = ""
    if len(node.args) >= 2:
        val = _resolve_constant(node.args[1])
        if val is not None:
            pv_suffix = str(val)

    # Keyword arguments
    kind = ""
    string = False
    for kw in node.keywords:
        if kw.arg == "kind":
            val = _resolve_constant(kw.value)
            if val is not None:
                kind = str(val)
        elif kw.arg == "string":
            val = _resolve_constant(kw.value)
            if val is not None:
                string = bool(val)

    # Detect if this is a sub-device (not an EpicsSignal variant)
    is_subdevice = signal_class not in (
        "EpicsSignal",
        "EpicsSignalRO",
        "EpicsSignalWithRBV",
        "EpicsSignalNoValidation",
        "Signal",
        "InternalSignal",
    )

    return ComponentInfo(
        attr_name="",  # filled by caller
        comp_type=comp_type,
        signal_class=signal_class,
        pv_suffix=pv_suffix,
        kind=kind,
        string=string,
        is_subdevice=is_subdevice,
    )


def parse_device_file(filepath: Path, module_path: str) -> list[DeviceClassInfo]:
    """AST-parse a single Python file for ophyd Device classes."""
    try:
        tree = ast.parse(filepath.read_text())
    except SyntaxError as exc:
        print(f"  Warning: syntax error in {filepath}: {exc}")
        return []

    classes: list[DeviceClassInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)

        # Collect components
        components: list[ComponentInfo] = []
        for item in node.body:
            if not isinstance(item, ast.Assign):
                continue
            if len(item.targets) != 1:
                continue
            target = item.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(item.value, ast.Call):
                continue

            comp = _parse_component_call(item.value)
            if comp is not None:
                comp.attr_name = target.id
                components.append(comp)

        if not components and not bases:
            continue

        # Only include classes that look like Device subclasses
        device_like_bases = {
            "Device",
            "TomoScanDevice",
            "TomoScanPSODevice",
            "TomoScanHelicalDevice",
            "TomoScan2BMDevice",
            "MCTOpticsLensInfo",
            "MCTOpticsLensOffset",
            "MCTOpticsLensControl",
            "MCTOpticsCameraControl",
            "MCTOptics",
        }
        if not components and not any(b in device_like_bases for b in bases):
            continue

        docstring = ast.get_docstring(node) or ""

        classes.append(
            DeviceClassInfo(
                name=node.name,
                module=module_path,
                docstring=docstring,
                bases=bases,
                components=components,
            )
        )

    return classes


def discover_device_classes(src_dir: Path) -> list[DeviceClassInfo]:
    """Walk src_dir for all devices/ directories and parse device classes."""
    all_classes: list[DeviceClassInfo] = []

    for devices_dir in sorted(src_dir.rglob("devices")):
        if not devices_dir.is_dir():
            continue
        for py_file in sorted(devices_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            # Build a dotted module path from src_dir
            rel = py_file.relative_to(src_dir)
            module_path = str(rel.with_suffix("")).replace("/", ".")

            classes = parse_device_file(py_file, module_path)
            all_classes.extend(classes)

    return all_classes


# ---------------------------------------------------------------------------
# Inheritance resolution
# ---------------------------------------------------------------------------


def resolve_inheritance(
    classes: list[DeviceClassInfo],
) -> dict[str, list[ComponentInfo]]:
    """For each class, build the full component list including inherited ones.

    Returns a dict mapping class name -> list of all components (inherited first,
    then own, with overrides applied).
    """
    by_name: dict[str, DeviceClassInfo] = {c.name: c for c in classes}
    cache: dict[str, list[ComponentInfo]] = {}

    def _get_all(cls_name: str) -> list[ComponentInfo]:
        if cls_name in cache:
            return cache[cls_name]

        cls = by_name.get(cls_name)
        if cls is None:
            return []

        # Collect inherited components
        inherited: dict[str, ComponentInfo] = {}
        for base in cls.bases:
            for comp in _get_all(base):
                inherited[comp.attr_name] = comp

        # Apply own components (overrides)
        for comp in cls.components:
            inherited[comp.attr_name] = comp

        result = list(inherited.values())
        cache[cls_name] = result
        return result

    for cls in classes:
        _get_all(cls.name)

    return cache


def _get_inheritance_chain(
    cls_name: str, by_name: dict[str, DeviceClassInfo]
) -> list[str]:
    """Build the inheritance chain for a class."""
    chain = [cls_name]
    current = cls_name
    while current in by_name:
        bases = by_name[current].bases
        if not bases:
            break
        parent = bases[0]  # primary base
        chain.append(parent)
        current = parent
    return chain


# ---------------------------------------------------------------------------
# Plan source collection
# ---------------------------------------------------------------------------


def collect_plan_sources(src_dir: Path) -> dict[str, str]:
    """Walk all plans/ directories under src_dir and read each .py file.

    Returns a dict mapping module_path -> source_code.
    """
    sources: dict[str, str] = {}
    for plans_dir in sorted(src_dir.rglob("plans")):
        if not plans_dir.is_dir():
            continue
        for py_file in sorted(plans_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            rel = py_file.relative_to(src_dir)
            module_path = str(rel.with_suffix("")).replace("/", ".")
            try:
                sources[module_path] = py_file.read_text()
            except OSError as exc:
                print(f"  Warning: could not read {py_file}: {exc}")
    return sources


# ---------------------------------------------------------------------------
# LLM-assisted generation helpers
# ---------------------------------------------------------------------------


def format_device_data_for_llm(
    instances: list[DeviceInstance],
    classes: list[DeviceClassInfo],
    all_components: dict[str, list[ComponentInfo]],
) -> str:
    """Serialize parsed device structures to JSON for LLM consumption."""
    by_name = {c.name: c for c in classes}

    instances_data = [
        {
            "name": inst.name,
            "class_path": inst.class_path,
            "class_short": inst.class_path.rsplit(".", 1)[-1],
            "prefix": inst.prefix,
            "labels": inst.labels,
            "kwargs": inst.kwargs,
        }
        for inst in instances
    ]

    classes_data = []
    for cls in classes:
        chain = _get_inheritance_chain(cls.name, by_name)
        comps = all_components.get(cls.name, cls.components)
        classes_data.append(
            {
                "name": cls.name,
                "module": cls.module,
                "docstring": cls.docstring,
                "inheritance_chain": chain,
                "components": [
                    {
                        "attr_name": c.attr_name,
                        "comp_type": c.comp_type,
                        "signal_class": c.signal_class,
                        "pv_suffix": c.pv_suffix,
                        "kind": c.kind,
                        "string": c.string,
                        "is_subdevice": c.is_subdevice,
                    }
                    for c in comps
                ],
            }
        )

    return json.dumps(
        {"instances": instances_data, "classes": classes_data},
        indent=2,
    )


def build_skills_llm_config(config: BaitConfig):
    """Build an AG2 LLMConfig for skills generation.

    Delegates to the shared builder in utils.
    """
    return build_llm_config(config)


# ---------------------------------------------------------------------------
# System prompt for LLM-assisted generation
# ---------------------------------------------------------------------------

SKILLS_SYSTEM_PROMPT = """\
You are an expert technical writer for ophyd/Bluesky beamline control systems.
You will receive structured JSON data describing device instances, device classes
(with full inherited components), plan source code, and optionally a startup.py
file. Your job is to generate a comprehensive `device_skills.md` markdown
document that an AI agent can use to understand and interact with these devices.

Generate EXACTLY these 5 sections:

## 1. Device Instances
List every device from devices.yml. For each device, include its name, class,
EPICS prefix, and labels. Add a brief purpose description inferred from the
class name, docstring, and prefix.

## 2. Device Class Reference
For each class:
- Show the full inheritance chain.
- Group components semantically (e.g., "Rotation Parameters", "Dark Field
  Config", "PSO Settings") rather than listing them flat.
- For each component, show: attribute name, signal class, PV suffix, kind.
- Add practical notes on which signals are commonly read vs. written, inferred
  from plan source code (look for `.get()`, `.put()`, `bps.rd()`, `bps.mv()`
  calls on these attributes).

## 3. Common Workflows
Explain how to actually use these devices in practice:
- How to run a scan (using the RunEngine and plans)
- How to check device status
- How to configure scan parameters before acquisition
- How to monitor during acquisition
Use concrete examples with actual device names and prefixes from the data.
If plan source code is provided, reference the actual plan functions and
their parameters.

## 4. Signal Reference Guide
- Explain `.get()` / `.put()` usage for interactive use.
- Explain PV address resolution: full PV = prefix + suffix.
- Explain `kind` meanings (config, normal, omitted).
- Explain `bps.rd()` / `bps.mv()` for use within plans.
- Provide concrete examples using actual device names and prefixes.

## 5. Known Limitations
- Note any commented-out features visible in the class source.
- Note scan types or device capabilities that appear incomplete.
- Note any sub-devices that are referenced but not defined in the parsed data.

CRITICAL RULES:
- Do NOT invent PV names, device features, or plan functions not present in the
  source data.
- Use markdown formatting suitable for AI agent consumption.
- Start the document with a top-level heading "# Device Skills Reference" and
  an auto-generated notice.
- Be thorough but concise — include all components, but group and annotate them
  rather than just listing raw tables.
"""


def _build_user_message(
    device_json: str,
    plan_sources: dict[str, str],
    startup_source: str,
) -> str:
    """Build the user message containing all data for the LLM."""
    parts = [
        "Generate a comprehensive device_skills.md document from the following data.\n",
        "## Device Data (JSON)\n",
        f"```json\n{device_json}\n```\n",
    ]

    if plan_sources:
        parts.append("## Plan Source Code\n")
        for module_path, source in plan_sources.items():
            parts.append(f"### `{module_path}`\n")
            parts.append(f"```python\n{source}\n```\n")

    if startup_source:
        parts.append("## startup.py\n")
        parts.append(f"```python\n{startup_source}\n```\n")

    return "\n".join(parts)


def generate_markdown_llm(
    instances: list[DeviceInstance],
    classes: list[DeviceClassInfo],
    all_components: dict[str, list[ComponentInfo]],
    plan_sources: dict[str, str],
    startup_source: str,
    config: BaitConfig,
) -> str:
    """Generate device_skills.md using an AG2 agent chat (LLM-assisted)."""
    llm_config = build_skills_llm_config(config)
    device_json = format_device_data_for_llm(instances, classes, all_components)

    # Agent that generates the document
    skills_writer = autogen.AssistantAgent(
        "skills_writer",
        llm_config=llm_config,
        system_message=SKILLS_SYSTEM_PROMPT,
    )

    # Agent that sends the data and collects response
    requester = autogen.UserProxyAgent(
        "requester",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: True,
        code_execution_config=False,
    )

    user_message = _build_user_message(device_json, plan_sources, startup_source)

    print("Sending device data to LLM for skills generation...")
    chat_result = requester.initiate_chat(
        recipient=skills_writer,
        message=user_message,
    )

    result = chat_result.summary
    if not result:
        raise RuntimeError("LLM returned an empty response.")
    return result


# ---------------------------------------------------------------------------
# Static markdown generation (fallback)
# ---------------------------------------------------------------------------


def generate_markdown_static(
    instances: list[DeviceInstance],
    classes: list[DeviceClassInfo],
    all_components: dict[str, list[ComponentInfo]],
    bits_path: Path,
    package_name: str,
) -> str:
    """Generate the complete device_skills.md content (static/template-based)."""
    by_name = {c.name: c for c in classes}
    lines: list[str] = []

    # Header
    lines.append("# Device Skills Reference")
    lines.append("")
    lines.append(
        "> **Auto-generated** by `init-bits-skills`. "
        "Do not edit manually — re-run the command to update."
    )
    lines.append("")
    lines.append(
        "This file describes all ophyd device classes and instantiated "
        "devices in this BITS instrument package. It is designed to be "
        "consumed by AI agents that need to understand and interact with "
        "the beamline's hardware."
    )
    lines.append("")

    # ----- Device Instances -----
    lines.append("## Device Instances")
    lines.append("")
    lines.append("These devices are instantiated at startup via `devices.yml`:")
    lines.append("")

    if instances:
        lines.append("| Name | Class | Prefix | Labels |")
        lines.append("|------|-------|--------|--------|")
        for inst in instances:
            labels = ", ".join(inst.labels) if inst.labels else ""
            cls_short = inst.class_path.rsplit(".", 1)[-1]
            lines.append(
                f"| `{inst.name}` | `{cls_short}` | `{inst.prefix}` | {labels} |"
            )
        lines.append("")
    else:
        lines.append("_No device instances found in devices.yml._")
        lines.append("")

    # ----- Device Class Reference -----
    lines.append("## Device Class Reference")
    lines.append("")

    for cls in classes:
        lines.append(f"### `{cls.name}`")
        lines.append("")
        lines.append(f"**Module:** `{cls.module}`")
        lines.append("")

        # Inheritance chain
        chain = _get_inheritance_chain(cls.name, by_name)
        if len(chain) > 1:
            lines.append("**Inheritance:** " + " → ".join(f"`{c}`" for c in chain))
            lines.append("")

        # Docstring (first paragraph only)
        if cls.docstring:
            first_para = cls.docstring.split("\n\n")[0].strip()
            lines.append(first_para)
            lines.append("")

        # Component table — use full inherited set for leaf/important classes
        comps = all_components.get(cls.name, cls.components)
        if comps:
            lines.append("| Component | Type | PV Suffix | Kind | Notes |")
            lines.append("|-----------|------|-----------|------|-------|")
            for comp in comps:
                notes_parts = []
                if comp.string:
                    notes_parts.append("string=True")
                if comp.is_subdevice:
                    notes_parts.append(f"sub-device: {comp.signal_class}")
                if comp.comp_type == "FCpt":
                    notes_parts.append("formatted")
                notes = ", ".join(notes_parts)

                signal_display = comp.signal_class
                lines.append(
                    f"| `{comp.attr_name}` | {signal_display} "
                    f"| `{comp.pv_suffix}` | {comp.kind} | {notes} |"
                )
            lines.append("")
        else:
            lines.append("_No components defined._")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ----- Interaction Guide -----
    lines.append("## Interaction Guide")
    lines.append("")
    lines.append("### Reading and Writing Signals")
    lines.append("")
    lines.append("```python")
    lines.append("# Read a signal value")
    lines.append("value = device.component_name.get()")
    lines.append("")
    lines.append("# Write a signal value (EpicsSignal only, not EpicsSignalRO)")
    lines.append("device.component_name.put(new_value)")
    lines.append("```")
    lines.append("")

    # Use a real example — prefer an instance with a prefix
    prefixed = [i for i in instances if i.prefix]
    if prefixed or instances:
        inst = prefixed[0] if prefixed else instances[0]
        lines.append("**Example with instantiated device:**")
        lines.append("")
        lines.append("```python")
        lines.append(f"# Read rotation start from '{inst.name}'")
        lines.append(f"angle = {inst.name}.rotation_start.get()")
        lines.append("")
        lines.append("# Set exposure time")
        lines.append(f"{inst.name}.exposure_time.put(0.5)")
        lines.append("```")
        lines.append("")

    lines.append("### Signal Types")
    lines.append("")
    lines.append("| Signal Class | Read | Write | Description |")
    lines.append("|-------------|------|-------|-------------|")
    lines.append("| `EpicsSignal` | `.get()` | `.put(value)` | Read/write EPICS PV |")
    lines.append("| `EpicsSignalRO` | `.get()` | _read-only_ | Read-only EPICS PV |")
    lines.append(
        "| `EpicsSignalWithRBV` | `.get()` | `.put(value)` | "
        "Read/write with readback verification |"
    )
    lines.append("")

    lines.append("### PV Address Resolution")
    lines.append("")
    lines.append("The full EPICS PV address is: **`prefix` + `PV suffix`**")
    lines.append("")
    if prefixed or instances:
        inst = prefixed[0] if prefixed else instances[0]
        lines.append(f"For the `{inst.name}` device (prefix `{inst.prefix}`):")
        lines.append("")
        lines.append(
            f"- `{inst.name}.rotation_start` → PV `{inst.prefix}RotationStart`"
        )
        lines.append(f"- `{inst.name}.scan_status` → PV `{inst.prefix}ScanStatus`")
        lines.append("")

    lines.append("### Kind Meanings")
    lines.append("")
    lines.append("| Kind | Meaning |")
    lines.append("|------|---------|")
    lines.append("| `config` | Saved with scan metadata (configuration values) |")
    lines.append("| `normal` | Read at each scan step (monitored during acquisition) |")
    lines.append("| `omitted` | Not recorded in scan data (control-only signals) |")
    lines.append('| _(empty)_ | Default ophyd behavior (typically "normal") |')
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Entry point for the init-bits-skills CLI command."""
    config = BaitConfig()

    if not config.bits.enabled:
        print("BITS integration is not enabled in config.yaml.")
        print("Set bits.enabled to true and provide bits.path.")
        sys.exit(1)

    bits_path = Path(config.bits.path)
    if not bits_path.is_dir():
        print(f"Error: BITS path does not exist: {bits_path}")
        sys.exit(1)

    src_dir = bits_path / config.bits.src_subdir
    if not src_dir.is_dir():
        print(f"Error: src directory does not exist: {src_dir}")
        sys.exit(1)

    package_name = config.bits.package_name
    configs_dir = src_dir / package_name / "configs"

    print(f"BITS path: {bits_path}")
    print(f"Package:   {package_name}")
    print(f"Configs:   {configs_dir}")
    print()

    # 1. Parse device YAML configs
    instances: list[DeviceInstance] = []
    if configs_dir.is_dir():
        instances = parse_devices_yaml(configs_dir)
        print(f"Found {len(instances)} device instance(s) in YAML configs")
    else:
        print(f"Warning: configs directory not found: {configs_dir}")

    # 2. AST-parse device Python files
    classes = discover_device_classes(src_dir)
    print(f"Found {len(classes)} device class(es) via AST parsing")

    # 3. Resolve inheritance
    all_components = resolve_inheritance(classes)

    # 4. Collect plan sources
    plan_sources = collect_plan_sources(src_dir)
    print(f"Found {len(plan_sources)} plan source file(s)")

    # 5. Read startup.py if it exists
    startup_source = ""
    startup_path = src_dir / package_name / "startup.py"
    if startup_path.exists():
        startup_source = startup_path.read_text()
        print(f"Found startup.py: {startup_path}")

    # 6. Generate markdown (LLM with static fallback)
    use_static = "--static" in sys.argv
    if use_static:
        print("Using static (template-based) generation...")
        md_content = generate_markdown_static(
            instances, classes, all_components, bits_path, package_name
        )
    else:
        try:
            md_content = generate_markdown_llm(
                instances,
                classes,
                all_components,
                plan_sources,
                startup_source,
                config,
            )
        except Exception as exc:
            print(f"LLM generation failed: {exc}")
            print("Falling back to static generation...")
            md_content = generate_markdown_static(
                instances, classes, all_components, bits_path, package_name
            )

    # 7. Write output
    output_path = bits_path / "device_skills.md"
    output_path.write_text(md_content)
    print(f"\nGenerated: {output_path}")
    print(f"  {len(md_content)} characters, {md_content.count(chr(10))} lines")


if __name__ == "__main__":
    main()
