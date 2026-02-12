# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

import unittest
from pathlib import Path
import os


class TestFloorset(unittest.TestCase):

    cpupc_cmd: str
    netlist: str
    out_netlist: str
    data: str
    label: str
    names: str
    error_log: str
    files: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.cpupc_cmd = (
            "python "
            + str(Path(__file__).resolve().parents[2] / "tools" / "cpupc_tools.py")
            + " "
        )
        parent = Path(__file__).resolve().parent
        cls.netlist = str(parent / "netlist.yml")
        cls.out_netlist = str(parent / "output_netlist.yml")
        cls.data = str(parent / "data.pt")
        cls.label = str(parent / "label.pt")
        cls.names = str(parent / "names.yml")
        cls.error_log = str(parent / "error.log")
        cls.files = f"--data {cls.data} --label {cls.label} --names {cls.names}"

    @classmethod
    def tearDownClass(cls) -> None:
        for file in [
            cls.data,
            cls.label,
            cls.names,
            cls.out_netlist,
            cls.error_log,
        ]:
            if os.path.exists(file):
                os.remove(file)

    def test_a_writefloorset(self):
        cmd = f"{self.cpupc_cmd} writefloorset {self.files} {self.netlist}"
        result = os.system(cmd)
        self.assertEqual(result, 0)

    def test_b_readfloorset(self):
        cmd_read = (
            f"{self.cpupc_cmd} readfloorset {self.files} --netlist {self.out_netlist}"
        )
        result_read = os.system(cmd_read)
        self.assertEqual(result_read, 0)

    def test_c_validate(self):
        cmd_validate = f"{self.cpupc_cmd} validate {self.files} -err {self.error_log} "
        for opt, numerr in [
            ("--area", 1),
            ("--mib", 2),
            ("--cluster", 1),
            ("--overlap", 1),
            ("--boundary", 1),
            ("--all", 6),
        ]:
            cmd_opt = cmd_validate + opt
            result_validate = os.system(cmd_opt)
            self.assertEqual(
                result_validate,
                numerr,
                f"Validation with option {opt} should report {numerr} errors",
            )

        cmd_opt = cmd_validate + "--ar "
        for ar, numerr in [(1.5, 3), (2.3, 1), (2.5, 0)]:
            cmd_ar = cmd_opt + str(ar)
            result_ar = os.system(cmd_ar)
            self.assertEqual(
                result_ar,
                numerr,
                f"Validation with aspect ratio {ar} should report {numerr} errors",
            )
