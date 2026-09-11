"""Tests for tool/pack_update.py, the package assembler.

Standard library only, like the tool. The suite fabricates package
directories in the shape seeml-update-compile emits (a plan file with the
SEEU magic, a build.sh) so no compiler build is needed; the one test that
assembles the stub for real is skipped when no C++ driver is on PATH.

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import io
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tool"))

import pack_update as pu  # noqa: E402

# The pre-P1 emitter's build.sh, abridged to the lines the packer touches.
LEGACY_BUILD_SH = (
    "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\nCXX=\"${CXX:-c++}\"\n"
    "FLAGS=\"-std=c++23 -O2 -Wall -Wextra -pthread -I. $TILE_FLAGS\"\n"
    "$CXX $FLAGS -c update_plan_embedded.cc -o update_plan_embedded.o\n"
    "$CXX $FLAGS -c update_main.cc -o update_main.o\n"
    "$CXX $FLAGS update_main.o update_plan_embedded.o -o model_update\n")

# A post-P1 emitter's build.sh branches on the stub itself.
MODERN_BUILD_SH = (
    "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\nCXX=\"${CXX:-c++}\"\n"
    "if [ -f update_plan_embedded.S ]; then\n"
    "  $CXX -c update_plan_embedded.S -o update_plan_embedded.o\n"
    "elif [ -f update_plan_embedded.cc ]; then\n"
    "  $CXX $FLAGS -c update_plan_embedded.cc -o update_plan_embedded.o\n"
    "else\n  exit 1\nfi\n")


def which_cxx():
    return os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") \
        or shutil.which("g++")


class PackageDir:
    """A fabricated package directory."""

    def __init__(self, build_sh=MODERN_BUILD_SH, plan=b"SEEU" + bytes(range(256)) * 3,
                 decimal_tu=True):
        self.dir = tempfile.mkdtemp(prefix="pack_update_test-")
        self.plan = plan
        with open(os.path.join(self.dir, pu.PLAN_FILE), "wb") as f:
            f.write(plan)
        with open(os.path.join(self.dir, pu.BUILD_SCRIPT), "w") as f:
            f.write(build_sh)
        os.chmod(os.path.join(self.dir, pu.BUILD_SCRIPT), 0o755)
        if decimal_tu:
            with open(os.path.join(self.dir, pu.DECIMAL_TU), "w") as f:
                f.write("const unsigned char kSeemlUpdatePlan[] = {0};\n")

    def path(self, name):
        return os.path.join(self.dir, name)

    def read(self, name, mode="r"):
        with open(self.path(name), mode) as f:
            return f.read()

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def run_main(argv):
    err = io.StringIO()
    with redirect_stderr(err):
        rc = pu.main(argv)
    return rc, err.getvalue()


class StubTest(unittest.TestCase):
    def test_declares_the_drivers_symbols_page_aligned(self):
        stub = pu.render_stub("update_plan.seeu", 16384)
        self.assertIn('.incbin "update_plan.seeu"', stub)
        self.assertIn(".balign 16384", stub)
        self.assertIn("SEEML_SYM(kSeemlUpdatePlan):", stub)
        self.assertIn("SEEML_SYM(kSeemlUpdatePlanSize):", stub)
        # The size is a label difference, never a literal that can go stale.
        self.assertRegex(stub, r"\.quad SEEML_SYM\(kSeemlUpdatePlan_end\) - "
                               r"SEEML_SYM\(kSeemlUpdatePlan\)\n")
        self.assertNotRegex(stub, r"\.(quad|long)\s+\d")
        self.assertIn("__USER_LABEL_PREFIX__", stub)
        # Both symbols are exported; the end label is not.
        self.assertIn(".globl SEEML_SYM(kSeemlUpdatePlan)\n", stub)
        self.assertIn(".globl SEEML_SYM(kSeemlUpdatePlanSize)\n", stub)
        self.assertNotIn(".globl SEEML_SYM(kSeemlUpdatePlan_end)", stub)
        # Only C comments: the preprocessor strips them on every target,
        # while the assembler's own comment character differs per ISA.
        self.assertNotIn("//", stub.replace("__TEXT,__const", ""))

    def test_refuses_a_plan_path_with_directories_or_quotes(self):
        with self.assertRaises(ValueError):
            pu.render_stub("sub/update_plan.seeu", 4096)
        with self.assertRaises(ValueError):
            pu.render_stub('bad".seeu', 4096)

    @unittest.skipIf(which_cxx() is None, "no C++ driver on PATH")
    def test_assembles_links_and_matches_the_plan_bytes(self):
        """The contract, end to end on this host's toolchain: the stub
        assembles under the C++ driver, links against the exact extern
        declarations update_main.cc uses, the symbol is page-aligned, and
        the bytes equal the file's — for an odd-sized plan, so the size
        symbol cannot be satisfied by padding."""
        pkg = PackageDir(plan=b"SEEU" + os.urandom(100003 - 4))
        self.addCleanup(pkg.cleanup)
        with open(pkg.path(pu.STUB_FILE), "w") as f:
            f.write(pu.render_stub(pu.PLAN_FILE, 16384))
        probe = r"""
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
extern const unsigned char kSeemlUpdatePlan[];
extern const size_t kSeemlUpdatePlanSize;
int main() {
  std::ifstream f("update_plan.seeu", std::ios::binary);
  std::string want((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  const bool same = want.size() == kSeemlUpdatePlanSize &&
      std::memcmp(want.data(), kSeemlUpdatePlan, want.size()) == 0;
  std::printf("%zu %zu %d\n", kSeemlUpdatePlanSize,
              (size_t)(reinterpret_cast<uintptr_t>(kSeemlUpdatePlan) % 16384),
              same ? 1 : 0);
  return same ? 0 : 1;
}
"""
        with open(pkg.path("probe.cc"), "w") as f:
            f.write(probe)
        cxx = which_cxx()
        # Exactly the build.sh recipe: the stub takes no C++ flags.
        subprocess.run([cxx, "-c", pu.STUB_FILE, "-o", "stub.o"],
                       cwd=pkg.dir, check=True)
        subprocess.run([cxx, "-std=c++23", "probe.cc", "stub.o", "-o", "probe"],
                       cwd=pkg.dir, check=True)
        out = subprocess.run(["./probe"], cwd=pkg.dir, check=True,
                             capture_output=True, text=True).stdout.split()
        self.assertEqual(out, [str(len(pkg.plan)), "0", "1"])


class PlanCheckTest(unittest.TestCase):
    def test_accepts_a_plan_and_returns_its_size(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        self.assertEqual(pu.check_plan(pkg.path(pu.PLAN_FILE)), len(pkg.plan))

    def test_rejects_missing_empty_and_foreign_files(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        with self.assertRaisesRegex(pu.PackError, "cannot read plan"):
            pu.check_plan(pkg.path("absent.seeu"))
        open(pkg.path("empty.seeu"), "wb").close()
        with self.assertRaisesRegex(pu.PackError, "is empty"):
            pu.check_plan(pkg.path("empty.seeu"))
        with open(pkg.path("model.smf"), "wb") as f:
            f.write(b"SMF1" + bytes(64))
        with self.assertRaisesRegex(pu.PackError, "plan magic"):
            pu.check_plan(pkg.path("model.smf"))


class BuildScriptTest(unittest.TestCase):
    def test_rewrites_only_the_legacy_compile_line(self):
        out = pu.adapt_build_script(LEGACY_BUILD_SH)
        self.assertIsNotNone(out)
        self.assertNotIn("-c update_plan_embedded.cc", out)
        self.assertIn(pu.STUB_COMPILE_LINE, out)
        # Everything else — including the link line naming the .o — stays.
        self.assertIn("update_main.o update_plan_embedded.o -o model_update", out)
        self.assertEqual(out.count("\n"), LEGACY_BUILD_SH.count("\n"))

    def test_leaves_a_stub_aware_script_alone(self):
        self.assertIsNone(pu.adapt_build_script(MODERN_BUILD_SH))

    def test_refuses_a_script_it_does_not_recognize(self):
        with self.assertRaisesRegex(pu.PackError, "not a seeml-update-compile"):
            pu.adapt_build_script("#!/bin/sh\nmake\n")
        # Two compile lines is not the emitter's script either.
        with self.assertRaisesRegex(pu.PackError, "refusing"):
            pu.adapt_build_script(LEGACY_BUILD_SH + pu.LEGACY_COMPILE_LINE)


class AtomicWriteTest(unittest.TestCase):
    def test_replaces_the_target_and_leaves_no_temp_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        target = os.path.join(d, "out.txt")
        with open(target, "w") as f:
            f.write("old")
        pu.write_atomic(target, b"new", executable=True)
        with open(target) as f:
            self.assertEqual(f.read(), "new")
        self.assertTrue(os.stat(target).st_mode & stat.S_IXUSR)
        self.assertEqual(os.listdir(d), ["out.txt"])

    def test_files_get_the_ordinary_create_mode_not_mkstemps_0600(self):
        """A package is copied to and built on other machines and by other
        users: the stub must be world-readable under a normal umask."""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        pu.write_atomic(os.path.join(d, "a.S"), b"x")
        pu.write_atomic(os.path.join(d, "b.sh"), b"x", executable=True)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(d, "a.S")).st_mode), 0o644)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(d, "b.sh")).st_mode), 0o755)


class PackTest(unittest.TestCase):
    def test_embeds_and_removes_the_decimal_tu(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir])
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.exists(pkg.path(pu.STUB_FILE)))
        self.assertFalse(os.path.exists(pkg.path(pu.DECIMAL_TU)))
        self.assertIn("removed update_plan_embedded.cc", err)
        # A stub-aware build.sh is untouched, byte for byte.
        self.assertEqual(pkg.read(pu.BUILD_SCRIPT), MODERN_BUILD_SH)
        # The plan itself is never rewritten.
        self.assertEqual(pkg.read(pu.PLAN_FILE, "rb"), pkg.plan)

    def test_keep_decimal_tu_flag(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        rc, _ = run_main([pkg.dir, "--keep-decimal-tu"])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(pkg.path(pu.DECIMAL_TU)))

    def test_adapts_a_legacy_package(self):
        pkg = PackageDir(build_sh=LEGACY_BUILD_SH)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir])
        self.assertEqual(rc, 0, err)
        self.assertIn("rewrote build.sh", err)
        script = pkg.read(pu.BUILD_SCRIPT)
        self.assertIn(pu.STUB_COMPILE_LINE, script)
        self.assertTrue(os.stat(pkg.path(pu.BUILD_SCRIPT)).st_mode & stat.S_IXUSR)

    def test_is_idempotent(self):
        pkg = PackageDir(build_sh=LEGACY_BUILD_SH)
        self.addCleanup(pkg.cleanup)
        self.assertEqual(run_main([pkg.dir])[0], 0)
        first = pkg.read(pu.STUB_FILE), pkg.read(pu.BUILD_SCRIPT)
        self.assertEqual(run_main([pkg.dir])[0], 0)
        self.assertEqual((pkg.read(pu.STUB_FILE), pkg.read(pu.BUILD_SCRIPT)), first)

    def test_page_align_reaches_the_stub(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        self.assertEqual(run_main([pkg.dir, "--page-align", "4096"])[0], 0)
        self.assertIn(".balign 4096", pkg.read(pu.STUB_FILE))

    def test_missing_plan_or_script_is_exit_1(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        os.unlink(pkg.path(pu.PLAN_FILE))
        rc, err = run_main([pkg.dir])
        self.assertEqual(rc, 1)
        self.assertIn("cannot read plan", err)
        pkg2 = PackageDir()
        self.addCleanup(pkg2.cleanup)
        os.unlink(pkg2.path(pu.BUILD_SCRIPT))
        rc, err = run_main([pkg2.dir])
        self.assertEqual(rc, 1)
        self.assertIn("build.sh", err)
        # The failure happened before anything was written.
        self.assertFalse(os.path.exists(pkg2.path(pu.STUB_FILE)))

    def test_build_runs_the_packages_script_with_cxx(self):
        script = ("#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\n"
                  "if [ -f update_plan_embedded.S ]; then :; fi\n"
                  "printf '%s' \"$CXX\" > cxx.txt\n"
                  "printf '#!/bin/sh\\n' > model_update\nchmod +x model_update\n")
        pkg = PackageDir(build_sh=script)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir, "--build", "--cxx", "my-cross-c++",
                            "--report", pkg.path("pack.json")])
        self.assertEqual(rc, 0, err)
        self.assertEqual(pkg.read("cxx.txt"), "my-cross-c++")
        self.assertIn("built", err)
        import json
        report = json.loads(pkg.read("pack.json"))
        self.assertEqual(report["schema"], 1)
        self.assertTrue(report["built"])
        self.assertEqual(report["plan_bytes"], len(pkg.plan))
        self.assertIsInstance(report["build_seconds"], float)

    def test_failing_build_is_exit_1(self):
        script = ("#!/bin/sh\nif [ -f update_plan_embedded.S ]; then :; fi\n"
                  "exit 7\n")
        pkg = PackageDir(build_sh=script)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir, "--build"])
        self.assertEqual(rc, 1)
        self.assertIn("exit 7", err)

    def test_build_that_leaves_no_binary_is_exit_1(self):
        script = "#!/bin/sh\nif [ -f update_plan_embedded.S ]; then :; fi\n"
        pkg = PackageDir(build_sh=script)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir, "--build"])
        self.assertEqual(rc, 1)
        self.assertIn("no executable", err)


class CliTest(unittest.TestCase):
    """Strict like every SeeML tool: exit 2, never a default."""

    def assert_exit_2(self, argv):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                pu.parse_args(argv)
        self.assertEqual(cm.exception.code, 2)

    def test_bad_arguments(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        self.assert_exit_2([])
        self.assert_exit_2([pkg.dir, "--bogus"])
        self.assert_exit_2([pkg.dir, "--bui"])           # no abbreviations
        self.assert_exit_2([pkg.dir, "--page-align", "1000"])
        self.assert_exit_2([pkg.dir, "--page-align", "32"])
        self.assert_exit_2([pkg.dir, "--page-align", "131072"])
        self.assert_exit_2([pkg.dir, "--page-align", "big"])
        self.assert_exit_2([pkg.dir, "--cxx"])
        self.assert_exit_2([pkg.dir, "--cxx", "--build"])
        self.assert_exit_2([pkg.path("absent")])
        self.assert_exit_2([pkg.path(pu.PLAN_FILE)])      # a file, not a dir

    def test_defaults(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        opts = pu.parse_args([pkg.dir])
        self.assertEqual(opts.page_align, 16384)
        self.assertFalse(opts.build)
        self.assertIsNone(opts.cxx)
        self.assertIsNone(opts.report)


if __name__ == "__main__":
    unittest.main()
