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
            resolved = capybara.Settings._resolve_engine(str(custom))
            self.assertEqual(resolved, custom)

    def test_resolve_engine_should_fall_back_to_path_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "llama-server"
            fake.write_bytes(b"x")
            empty_home = Path(tmp) / "empty-home"
            empty_home.mkdir()
            env = {"CAPYBARA_HOME": str(empty_home)}
            with mock.patch.dict(os.environ, env, clear=False), \
                 mock.patch("shutil.which", return_value=str(fake)):
                resolved = capybara.Settings._resolve_engine(None)
            self.assertEqual(resolved, fake)

    def test_resolve_engine_should_default_to_home_bin(self):
        home = Path("/nonexistent-capybara-home-for-tests")
        with mock.patch.dict(os.environ, {"CAPYBARA_HOME": str(home)}, clear=False), \
             mock.patch("shutil.which", return_value=None):
            resolved = capybara.Settings._resolve_engine(None)
        self.assertEqual(resolved, home / "bin" / "llama-server")


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


if __name__ == "__main__":
    unittest.main()
