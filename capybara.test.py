#!/usr/bin/env python3
"""Unit tests for capybara.py (pure logic; no network or engine needed)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capybara
import server


class ParseScalarTests(unittest.TestCase):
    def test_parse_scalar_should_cast_numbers_and_bools(self):
        self.assertEqual(capybara.parse_scalar(" 8 "), 8)
        self.assertEqual(capybara.parse_scalar("0.75"), 0.75)
        self.assertIs(capybara.parse_scalar("true"), True)
        self.assertIs(capybara.parse_scalar("False"), False)
        self.assertEqual(capybara.parse_scalar("hello"), "hello")

    def test_parse_scalar_should_strip_matching_quotes(self):
        self.assertEqual(capybara.parse_scalar('"~/.capybara/models"'), "~/.capybara/models")
        self.assertEqual(capybara.parse_scalar("'x'"), "x")


class ConfigTextTests(unittest.TestCase):
    def test_parse_config_text_should_build_nested_maps(self):
        text = """
# comment line
runtime:
  threads: 8
  gpu_layers: 999

server:
  host: 127.0.0.1
  port: 11434
models:
  directory: "~/.capybara/models"
"""
        cfg = capybara.parse_config_text(text)
        self.assertEqual(cfg["runtime"], {"threads": 8, "gpu_layers": 999})
        self.assertEqual(cfg["server"], {"host": "127.0.0.1", "port": 11434})
        self.assertEqual(cfg["models"]["directory"], "~/.capybara/models")

    def test_parse_config_text_should_return_empty_for_empty_input(self):
        self.assertEqual(capybara.parse_config_text(""), {})
        self.assertEqual(capybara.parse_config_text("# only a comment"), {})


class SettingsTests(unittest.TestCase):
    def _write_config(self, home: Path, body: str) -> None:
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(body)

    def test_load_settings_should_use_defaults_without_config(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            settings = capybara.load_settings(Path(tmp))
            self.assertEqual(settings.host, "127.0.0.1")
            self.assertEqual(settings.port, 11434)
            self.assertEqual(settings.models, Path(tmp) / "models")

    def test_load_settings_should_read_config_file(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            home = Path(tmp)
            self._write_config(home, "server:\n  port: 9999\nruntime:\n  threads: 3\n")
            settings = capybara.load_settings(home)
            self.assertEqual(settings.port, 9999)
            self.assertEqual(settings.threads, 3)
            self.assertEqual(settings.base_url, "http://127.0.0.1:9999")

    def test_load_settings_environment_overrides_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self._write_config(home, "server:\n  port: 9999\n")
            env = {"CAPYBARA_PORT": "1234", "CAPYBARA_HOST": "0.0.0.0"}
            with mock.patch.dict(os.environ, env, clear=False):
                settings = capybara.load_settings(home)
            self.assertEqual(settings.port, 1234)
            self.assertEqual(settings.host, "0.0.0.0")


class HfSpecTests(unittest.TestCase):
    def test_split_hf_spec_should_handle_all_shapes(self):
        self.assertEqual(capybara.split_hf_spec("owner/repo"), ("owner/repo", None))
        self.assertEqual(capybara.split_hf_spec("owner/repo:Q4_K_M"), ("owner/repo", "Q4_K_M"))
        self.assertIsNone(capybara.split_hf_spec("llama3"))
        self.assertIsNone(capybara.split_hf_spec("https://example.com/m.gguf"))

    def test_resolve_alias_should_expand_known_names(self):
        self.assertNotEqual(capybara.resolve_alias("LLaMA3"), "LLaMA3")
        self.assertEqual(capybara.resolve_alias("unknown-model"), "unknown-model")

    def test_resolve_family_should_use_default_variant(self):
        spec = capybara.resolve_alias("qwen3")
        self.assertTrue(spec.startswith("unsloth/Qwen3-8B-GGUF"), spec)

    def test_resolve_family_should_support_size_and_quant_suffixes(self):
        self.assertEqual(
            capybara.resolve_alias("qwen3:14b"),
            "unsloth/Qwen3-14B-GGUF")
        self.assertEqual(
            capybara.resolve_alias("qwen3:14b:q8_0"),
            "unsloth/Qwen3-14B-GGUF:q8_0")
        self.assertEqual(
            capybara.resolve_alias("deepseek-r1:32b"),
            "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF")
        self.assertEqual(
            capybara.resolve_alias("gpt-oss:20b"),
            "ggml-org/gpt-oss-20b-GGUF")

    def test_resolve_family_should_pass_through_unknown_variants(self):
        self.assertEqual(capybara.resolve_alias("qwen3:nosize"), "qwen3:nosize")
        self.assertEqual(capybara.resolve_alias("nosuch:1b"), "nosuch:1b")

    def test_model_families_should_be_wellformed(self):
        for family, variants in capybara.MODEL_FAMILIES.items():
            self.assertIn("default", variants, family)
            for name, spec in variants.items():
                self.assertRegex(spec, r"^[^/]+/[^/]+$", f"{family}:{name}")

    def test_agents_registry_should_be_complete(self):
        for name, spec in capybara.AGENTS.items():
            for key in ("github", "check", "install", "desc"):
                self.assertIn(key, spec, f"{name} missing {key}")
            self.assertNotIn(" ", name)

    def test_clean_model_id_should_strip_extension_and_quant(self):
        self.assertEqual(
            capybara.clean_model_id(Path("Qwen3-8B-Q4_K_M.gguf")),
            "qwen3-8b")

    def test_select_gguf_files_should_prefer_q4_k_m(self):
        files = ["m.Q8_0.gguf", "m.Q4_K_M.gguf", "m.Q2_K.gguf"]
        self.assertEqual(capybara.select_gguf_files(files), ["m.Q4_K_M.gguf"])

    def test_select_gguf_files_should_filter_by_quant(self):
        files = ["m.Q8_0.gguf", "m.Q2_K.gguf"]
        self.assertEqual(capybara.select_gguf_files(files, "Q2_K"), ["m.Q2_K.gguf"])

    def test_select_gguf_files_should_return_empty_when_quant_missing(self):
        self.assertEqual(capybara.select_gguf_files(["m.Q8_0.gguf"], "Q5_K_M"), [])

    def test_select_gguf_files_should_group_shards(self):
        files = ["m.Q4_K_M-00002-of-00002.gguf", "other.Q8_0.gguf",
                 "m.Q4_K_M-00001-of-00002.gguf"]
        chosen = capybara.select_gguf_files(files)
        self.assertEqual(chosen, ["m.Q4_K_M-00001-of-00002.gguf",
                                  "m.Q4_K_M-00002-of-00002.gguf"])

    def test_select_gguf_files_should_prefer_single_file_over_shards(self):
        files = ["s-00001-of-00002.gguf", "s-00002-of-00002.gguf", "single.Q4_K_M.gguf"]
        self.assertEqual(capybara.select_gguf_files(files), ["single.Q4_K_M.gguf"])


class ModelfileTests(unittest.TestCase):
    def test_parse_modelfile_should_read_directives(self):
        text = """
# a comment
FROM llama3
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM You are a capybara.
"""
        parsed = capybara.parse_modelfile(text)
        self.assertEqual(parsed["base"], "llama3")
        self.assertEqual(parsed["params"], {"temperature": 0.7, "num_ctx": 4096})
        self.assertEqual(parsed["system"], "You are a capybara.")
        self.assertIsNone(parsed["template"])

    def test_parse_modelfile_should_support_multiline_system(self):
        text = 'FROM base\nSYSTEM """line one\nline two"""\n'
        parsed = capybara.parse_modelfile(text)
        self.assertEqual(parsed["system"], "line one\nline two")

    def test_parse_modelfile_is_case_insensitive(self):
        parsed = capybara.parse_modelfile("from x\ntemplate {{ .Prompt }}")
        self.assertEqual(parsed["base"], "x")
        self.assertEqual(parsed["template"], "{{ .Prompt }}")


class ResolveModelTests(unittest.TestCase):
    def test_resolve_model_should_match_exact_stem_and_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir()
            (models / "Alpha-Q4_K_M.gguf").write_bytes(b"x")
            (models / "Big-00001-of-00002.gguf").write_bytes(b"x")
            (models / "Big-00002-of-00002.gguf").write_bytes(b"x")
            settings = capybara.Settings(Path(tmp), {})
            self.assertEqual(capybara.resolve_model(settings, "alpha"), models / "Alpha-Q4_K_M.gguf")
            self.assertEqual(capybara.resolve_model(settings, "ALPHA-Q4_K_M.GGUF").name.lower(),
                             "alpha-q4_k_m.gguf")
            self.assertEqual(capybara.resolve_model(settings, "Big"), models / "Big-00001-of-00002.gguf")
            self.assertIsNone(capybara.resolve_model(settings, "nope"))

    def test_resolve_model_should_accept_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp) / "elsewhere.gguf"
            external.write_bytes(b"x")
            settings = capybara.Settings(Path(tmp) / "home", {})
            self.assertEqual(capybara.resolve_model(settings, str(external)), external)


class ListHelpersTests(unittest.TestCase):
    def test_human_size_should_format_units(self):
        self.assertEqual(capybara.human_size(512), "512 B")
        self.assertEqual(capybara.human_size(2048), "2.0 KB")
        self.assertRegex(capybara.human_size(5 * 1024 ** 3), r"^5\.0 GB$")

    def test_group_models_should_collapse_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            (models / "b-00001-of-00002.gguf").write_bytes(b"12")
            (models / "b-00002-of-00002.gguf").write_bytes(b"34")
            (models / "solo.gguf").write_bytes(b"567")
            entries = dict(capybara.group_models(sorted(models.glob("*.gguf"))))
            self.assertEqual(entries, {"b": 4, "solo": 3})


class SidecarTests(unittest.TestCase):
    def test_sidecar_for_should_read_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"x")
            model.with_suffix(".capybara.json").write_text(
                json.dumps({"system": "be nice", "params": {"num_predict": 64}}))
            meta = capybara.sidecar_for(model)
            self.assertEqual(meta["system"], "be nice")

    def test_request_context_should_map_params_and_system(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"x")
            model.with_suffix(".capybara.json").write_text(json.dumps({
                "system": "you are a capybara",
                "params": {"temperature": 0.5, "num_ctx": 2048, "num_predict": 32},
            }))
            messages, params = capybara.request_context(model)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(params, {"max_tokens": 32, "temperature": 0.5})


class TrimHistoryTests(unittest.TestCase):
    def test_trim_history_should_keep_short_histories_untouched(self):
        history = [{"role": "user", "content": "hi"}]
        self.assertEqual(capybara.trim_history(history, 1000), history)

    def test_trim_history_should_drop_oldest_turns_first(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 50},
            {"role": "assistant", "content": "b" * 50},
            {"role": "user", "content": "c" * 10},
        ]
        trimmed = capybara.trim_history(history, 80)
        contents = [m["content"] for m in trimmed if m["role"] != "system"]
        self.assertEqual(contents, ["b" * 50, "c" * 10])
        self.assertEqual(trimmed[0]["role"], "system")

    def test_trim_history_should_trim_down_to_budget(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 50},
            {"role": "assistant", "content": "b" * 50},
            {"role": "user", "content": "c" * 10},
        ]
        trimmed = capybara.trim_history(history, 40)
        contents = [m["content"] for m in trimmed if m["role"] != "system"]
        self.assertEqual(contents, ["c" * 10])

    def test_trim_history_should_always_keep_system_messages(self):
        history = [
            {"role": "system", "content": "s" * 60},
            {"role": "user", "content": "u" * 200},
        ]
        trimmed = capybara.trim_history(history, 70)
        self.assertEqual([m["role"] for m in trimmed], ["system"])


class EngineDiscoveryTests(unittest.TestCase):
    def test_settings_should_expose_internal_engine_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = capybara.Settings(Path(tmp), {"server": {"port": 11434}})
            self.assertEqual(settings.engine_port, 11435)

    def test_resolve_engine_should_prefer_existing_configured_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom-llama-server"
            custom.write_bytes(b"x")
            resolved = capybara.Settings(Path(tmp), {})._resolve_engine(str(custom))
            self.assertEqual(resolved, custom)

    def test_resolve_engine_should_fall_back_to_path_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "llama-server"
            fake.write_bytes(b"x")
            empty_home = Path(tmp) / "empty-home"
            empty_home.mkdir()
            with mock.patch("shutil.which", return_value=str(fake)):
                resolved = capybara.Settings(empty_home, {})._resolve_engine(None)
            self.assertEqual(resolved, fake)

    def test_resolve_engine_should_default_to_script_dir_bin(self):
        """With no engine anywhere, point the error message at the
        highest-priority default (bundled location next to capybara.py)."""
        with tempfile.TemporaryDirectory() as tmp:
            appdir = (Path(tmp) / "app").resolve()
            home = Path(tmp) / "home"
            home.mkdir()
            with mock.patch.object(capybara, "__file__", str(appdir / "capybara.py")), \
                 mock.patch("shutil.which", return_value=None):
                resolved = capybara.Settings(home, {})._resolve_engine(None)
            self.assertEqual(resolved,
                             appdir / "bin" / f"llama-server{capybara.EXE}")
            self.assertFalse(resolved.exists())

    def test_resolve_engine_should_find_bundled_engine_next_to_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Resolve first: on macOS tempfile paths contain symlinks that
            # _resolve_engine normalises away.
            appdir = (Path(tmp) / "app").resolve()
            (appdir / "bin").mkdir(parents=True)
            (appdir / "bin" / f"llama-server{capybara.EXE}").write_bytes(b"x")
            empty_home = Path(tmp) / "home"
            empty_home.mkdir()
            with mock.patch.object(capybara, "__file__", str(appdir / "capybara.py")), \
                 mock.patch("shutil.which", return_value=None):
                resolved = capybara.Settings(empty_home, {})._resolve_engine(None)
            self.assertEqual(resolved,
                             appdir / "bin" / f"llama-server{capybara.EXE}")
            self.assertTrue(resolved.exists())


class StateSafetyTests(unittest.TestCase):
    def _settings(self, tmp: Path) -> capybara.Settings:
        return capybara.Settings(tmp, {})

    def test_state_is_ours_should_reject_missing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            self.assertFalse(capybara.state_is_ours(settings))

    def test_state_is_ours_should_check_process_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            settings.run_dir.mkdir(parents=True)
            settings.state_file.write_text(json.dumps({
                "mode": "gateway", "gateway_pid": 424242,
                "pid": 424243, "model": "x.gguf",
            }))
            with mock.patch.object(capybara, "process_comm", return_value="Python"):
                self.assertTrue(capybara.state_is_ours(settings))
            with mock.patch.object(capybara, "process_comm", return_value="vim"):
                self.assertFalse(capybara.state_is_ours(settings))

    def test_stop_server_should_not_kill_unrelated_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            settings.run_dir.mkdir(parents=True)
            settings.state_file.write_text(json.dumps({
                "mode": "gateway", "gateway_pid": 999001, "pid": 999002,
            }))
            killed = []
            with mock.patch.object(capybara, "process_comm", return_value="nginx"), \
                 mock.patch.object(capybara.os, "kill",
                                   side_effect=lambda p, s: killed.append((p, s))):
                capybara.stop_server(settings)
            self.assertEqual(killed, [])
            self.assertFalse(settings.state_file.exists())


class GatewaySidecarTests(unittest.TestCase):
    """Tests for the gateway module's request enrichment."""

    def _manager_with_model(self, tmp: Path) -> object:
        import server
        settings = capybara.Settings(tmp, {})
        mgr = server.EngineManager(settings)
        model = Path(tmp) / "m.gguf"
        model.write_bytes(b"x")
        model.with_suffix(".capybara.json").write_text(json.dumps({
            "system": "stay terse",
            "params": {"temperature": 0.2, "num_predict": 48},
        }))
        mgr.model = model
        return mgr

    def test_inject_sidecar_should_add_system_and_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager_with_model(Path(tmp))
            body = {"messages": [{"role": "user", "content": "hello"}]}
            out = server.inject_sidecar(mgr, body)
            self.assertEqual(out["messages"][0]["role"], "system")
            self.assertEqual(out["messages"][0]["content"], "stay terse")
            self.assertEqual(out["temperature"], 0.2)
            self.assertEqual(out["max_tokens"], 48)

    def test_inject_sidecar_should_respect_client_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager_with_model(Path(tmp))
            body = {
                "messages": [{"role": "system", "content": "client wins"},
                             {"role": "user", "content": "hi"}],
                "temperature": 1.5,
            }
            out = server.inject_sidecar(mgr, body)
            self.assertEqual(out["temperature"], 1.5)
            self.assertEqual(out["messages"][0]["content"], "client wins")


class OllamaRefTests(unittest.TestCase):
    def test_parse_ollama_ref(self):
        cases = {
            "llama3.2": ("llama3.2", "latest"),
            "gemma3:4b": ("gemma3", "4b"),
            "ollama/phi4": ("phi4", "latest"),
            "ol/qwen2.5:7b-instruct-q4_K_M": ("qwen2.5", "7b-instruct-q4_K_M"),
        }
        for spec, expected in cases.items():
            self.assertEqual(capybara.parse_ollama_ref(spec), expected, spec)
        for spec in ("http://x/y.gguf", "https://x/y.gguf",
                     "owner/repo", "owner/repo:Q4_K_M"):
            self.assertIsNone(capybara.parse_ollama_ref(spec), spec)

    def test_gguf_names(self):
        self.assertEqual(capybara.ollama_gguf_name("llama3.2", "latest"),
                         ["llama3.2.gguf"])
        self.assertEqual(capybara.ollama_gguf_name("gemma3", "4b"),
                         ["gemma3-4b.gguf"])
        self.assertEqual(
            capybara.ollama_gguf_name("bigmodel", "latest", 2),
            ["bigmodel-00001-of-00002.gguf", "bigmodel-00002-of-00002.gguf"])


class OllamaInstallTests(unittest.TestCase):
    MANIFEST = {
        "layers": [
            {"mediaType": capybara.OLLAMA_SYSTEM_MT,
             "digest": "sha256:sys", "size": 10},
            {"mediaType": capybara.OLLAMA_MODEL_MT,
             "digest": "sha256:model", "size": 100},
            {"mediaType": capybara.OLLAMA_PARAMS_MT,
             "digest": "sha256:params", "size": 30},
        ],
    }

    def _settings(self):
        return capybara.Settings(Path(tempfile.mkdtemp()), {})

    def test_install_writes_gguf_and_sidecar(self):
        settings = self._settings()
        blobs = {
            "sha256:model": b"GGUF-fake",
            "sha256:sys": b"You are terse.",
            "sha256:params": json.dumps({"temperature": 0.5,
                                         "num_gpu": 99, "stop": ["###"]}).encode(),
        }

        def fetch(digest, dest):
            dest.write_bytes(blobs[digest])

        def text(digest):
            return blobs[digest].decode("utf-8")

        result = capybara.install_ollama_manifest(
            settings, "tiny", "1b", dict(self.MANIFEST), fetch, text)
        self.assertEqual(result.name, "tiny-1b.gguf")
        gguf = settings.models / "tiny-1b.gguf"
        self.assertEqual(gguf.read_bytes(), b"GGUF-fake")
        meta = json.loads((settings.models / "tiny-1b.capybara.json").read_text())
        self.assertEqual(meta["system"], "You are terse.")
        self.assertEqual(meta["params"], {"temperature": 0.5, "stop": ["###"]})
        self.assertEqual(meta["base"], "ollama:tiny:1b")

    def test_import_local_hardlinks_existing_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            ollama_root = Path(tmp) / "models"
            blob_dir = ollama_root / "blobs"
            blob_dir.mkdir(parents=True)
            (blob_dir / "sha256-abc").write_bytes(b"GGUF-data")
            manifest_path = (ollama_root / "manifests" / "registry.ollama.ai"
                             / "library" / "mini" / "latest")
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({
                "layers": [{"mediaType": capybara.OLLAMA_MODEL_MT,
                            "digest": "sha256:abc", "size": 9}]}))
            settings = self._settings()
            result = capybara.import_ollama_local(settings, "mini", "latest",
                                                  root=ollama_root)
            self.assertIsNotNone(result)
            installed = settings.models / "mini.gguf"
            self.assertEqual(installed.read_bytes(), b"GGUF-data")
            # absent model returns None
            self.assertIsNone(capybara.import_ollama_local(
                settings, "nope", "latest", root=ollama_root))

    def test_pull_model_prefers_local_ollama_then_registry(self):
        settings = self._settings()
        calls = []

        def fake_import(s, name, tag, root=None):
            calls.append(("local", name, tag))
            return None

        def fake_registry(s, name, tag):
            calls.append(("registry", name, tag))
            return s.models / f"{name}.gguf"

        with mock.patch.object(capybara, "import_ollama_local", fake_import), \
             mock.patch.object(capybara, "pull_ollama_registry", fake_registry):
            result = capybara.pull_model(settings, "ollama/gemma3:4b")
        self.assertEqual(calls, [("local", "gemma3", "4b"),
                                 ("registry", "gemma3", "4b")])
        self.assertEqual(result, settings.models / "gemma3.gguf")


class DiskSpaceTests(unittest.TestCase):
    def test_require_free_space_dies_when_disk_cannot_hold_blob(self):
        dest = Path(tempfile.mkdtemp()) / "x.gguf"
        capybara.require_free_space(dest, 1024)  # real disk: plenty
        with mock.patch.object(capybara.shutil, "disk_usage",
                               return_value=mock.Mock(free=100)):
            with self.assertRaises(SystemExit):
                capybara.require_free_space(dest, 10_000_000_000)


class PortGuardTests(unittest.TestCase):
    def _settings(self, tmp: Path, port: int) -> capybara.Settings:
        return capybara.Settings(tmp, {"server": {"port": port}})

    def test_port_has_listener_should_detect_open_sockets(self):
        import socket
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                sock.listen(1)
                port = sock.getsockname()[1]
                settings = self._settings(Path(tmp), port)
                self.assertTrue(capybara.port_has_listener(settings))
            settings2 = self._settings(Path(tmp), 1)
            self.assertFalse(capybara.port_has_listener(settings2))


class KeepAliveTests(unittest.TestCase):
    def test_parse_keep_accepts_ollama_durations(self):
        cases = {
            None: 300.0,          # default
            "": 300.0,
            "5m": 300.0,
            "90": 90.0,
            "90s": 90.0,
            "2h": 7200.0,
            "1h30m": 5400.0,
            "500ms": 0.5,
            0: 0.0,
            -1: float("inf"),
            "-1": float("inf"),
            "garbage": 300.0,
            True: 300.0,
        }
        for value, expected in cases.items():
            self.assertEqual(capybara.parse_keep_alive(value), expected,
                             repr(value))

    def test_parse_keep_honours_custom_default(self):
        self.assertEqual(capybara.parse_keep_alive(None, default=42.0), 42.0)
        self.assertEqual(capybara.parse_keep_alive("bogus", default=7.0), 7.0)

    def test_settings_should_read_keep_alive_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.yaml").write_text("runtime:\n  keep_alive: 30s\n")
            settings = capybara.load_settings(home)
            self.assertEqual(settings.keep_alive_raw, "30s")


class OllamaOptionsTests(unittest.TestCase):
    def test_options_map_to_engine_params(self):
        out = capybara.ollama_options_to_openai({
            "temperature": 0.3, "num_predict": 64, "top_k": 40,
            "stop": "END", "repeat_penalty": 1.1, "seed": 9,
        })
        self.assertEqual(out, {"temperature": 0.3, "max_tokens": 64,
                               "top_k": 40, "stop": ["END"],
                               "repeat_penalty": 1.1, "seed": 9})

    def test_unmappable_and_unknown_options_are_dropped(self):
        out = capybara.ollama_options_to_openai({
            "num_ctx": 8192, "mirostat": 2, "wat": 1, "temperature": 0.5,
        })
        self.assertEqual(out, {"temperature": 0.5})

    def test_non_dict_input_is_ignored(self):
        self.assertEqual(capybara.ollama_options_to_openai(None), {})
        self.assertEqual(capybara.ollama_options_to_openai("x"), {})


class OllamaApiTests(unittest.TestCase):
    """End-to-end handler tests against a stubbed EngineManager."""

    def setUp(self) -> None:
        import server
        self.server = server

    def _manager(self, tmp: Path) -> Any:
        mgr = self.server.EngineManager(capybara.Settings(tmp, {}))
        return mgr

    def _request(self, method: str, path: str, body: bytes = b""):
        """Drive the Gateway handler without opening sockets."""
        import io
        from http.server import BaseHTTPRequestHandler

        captured: Dict[str, Any] = {}

        class Sink:
            def __init__(self) -> None:
                self.buffer = io.BytesIO()

            def write(self, data: bytes) -> int:
                return self.buffer.write(data)

            def flush(self) -> None:
                pass

            def getvalue(self) -> bytes:
                return self.buffer.getvalue()

        sink = Sink()
        handler = object.__new__(self.server.Gateway)
        handler.command = method
        handler.path = path
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = sink
        handler.close_connection = False

        def fake_send_response(code: int, *a: Any, **k: Any) -> None:
            captured["status"] = code

        def fake_send_header(k: str, v: str) -> None:
            captured.setdefault("headers", []).append((k, v))

        def fake_end_headers() -> None:
            pass

        handler.send_response = fake_send_response
        handler.send_header = fake_send_header
        handler.end_headers = fake_end_headers
        captured["handler"] = handler
        captured["sink"] = sink
        return handler, captured

    @staticmethod
    def _json_lines(raw: bytes) -> List[Dict[str, Any]]:
        import json as _json
        return [_json.loads(line) for line in raw.decode().splitlines() if line]

    def test_api_version_and_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(Path(tmp))
            (Path(tmp) / "models").mkdir()
            model = Path(tmp) / "models" / "tiny-1b.gguf"
            model.write_bytes(b"x")
            self.server.Gateway.manager = mgr
            for method, path in (("GET", "/api/version"), ("GET", "/api/tags")):
                handler, captured = self._request(method, path)
                handler.do_GET()
                payload = self._json_lines(captured["sink"].getvalue())[0]
                if path == "/api/version":
                    self.assertEqual(payload["version"], capybara.VERSION)
                else:
                    names = [m["name"] for m in payload["models"]]
                    self.assertIn("tiny-1b:latest", names)

    def test_api_chat_requires_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(Path(tmp))
            self.server.Gateway.manager = mgr
            handler, captured = self._request(
                "POST", "/api/chat", b'{"model":"x","messages":[]}')
            handler.api_chat()
            self.assertEqual(captured["status"], 400)
            payload = self._json_lines(captured["sink"].getvalue())[0]
            self.assertIn("messages", payload["error"])

    def test_api_chat_unknown_model_is_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(Path(tmp))
            (Path(tmp) / "models").mkdir(exist_ok=True)
            self.server.Gateway.manager = mgr
            handler, captured = self._request(
                "POST", "/api/chat",
                b'{"model":"nope","messages":[{"role":"user","content":"hi"}]}')
            handler.api_chat()
            self.assertEqual(captured["status"], 404)

    def test_api_generate_requires_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(Path(tmp))
            self.server.Gateway.manager = mgr
            handler, captured = self._request(
                "POST", "/api/generate", b'{"model":"x"}')
            handler.api_generate()
            self.assertEqual(captured["status"], 400)

    def test_usage_snapshot_reports_expires_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir(parents=True)
            model = models_dir / "tiny-1b-Q4_K_M.gguf"
            model.write_bytes(b"x")
            mgr = self._manager(Path(tmp))
            mgr.settings.models = models_dir
            mgr.model = model
            snap = mgr.usage_snapshot()
            self.assertEqual(len(snap["models"]), 1)
            entry = snap["models"][0]
            self.assertEqual(entry["details"]["quantization_level"], "Q4_K_M")
            self.assertTrue(entry["expires_at"])
            mgr.keep_alive = float("inf")
            self.assertIsNone(mgr.usage_snapshot()["models"][0]["expires_at"])

    def test_param_size_guess(self):
        self.assertEqual(self.server.param_size_guess("qwen-7B-instruct.gguf"), "7B")
        self.assertEqual(self.server.param_size_guess("SmolLM2-135M-Instruct.gguf"),
                         "135M")
        self.assertEqual(self.server.param_size_guess("mystery-model.gguf"), "")


if __name__ == "__main__":
    unittest.main()
