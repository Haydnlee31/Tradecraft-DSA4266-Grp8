"""Keep pip's shared requirements and editable-install metadata consistent."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyTests(unittest.TestCase):
    def test_shared_dependencies_match_package_metadata(self):
        requirements = {
            line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(requirements, set(project["dependencies"]))
        self.assertFalse(any("flwr[simulation]" in req for req in requirements))

    def test_simulation_extra_matches_optional_requirements(self):
        lines = {
            line.strip() for line in (ROOT / "requirements-simulation.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("-r requirements.txt", lines)
        lines.remove("-r requirements.txt")
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(lines, set(project["optional-dependencies"]["simulation"]))


if __name__ == "__main__":
    unittest.main()
