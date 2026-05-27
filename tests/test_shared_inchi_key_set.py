"""Tests for SharedInchiKeySet."""

from __future__ import annotations

import multiprocessing as mp
import unittest

from shallowtree.context.stock.shared_inchi_key_set import SharedInchiKeySet

KEY_A = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
KEY_B = "DDDDDDDDDDDDDD-EEEEEEEEEE-F"
KEY_C = "GGGGGGGGGGGGGG-HHHHHHHHHH-I"


def _child_attach_and_query(shm_name: str, count: int, queries: list, result_q):
    """Subprocess entry point: attach to the shared set and answer queries."""
    s = SharedInchiKeySet.attach(shm_name, count)
    try:
        result_q.put([q in s for q in queries])
    finally:
        s.close()


class TestSharedInchiKeySetBasics(unittest.TestCase):
    def test_build_and_membership(self):
        s = SharedInchiKeySet.build([KEY_B, KEY_A, KEY_C])
        try:
            self.assertEqual(len(s), 3)
            self.assertIn(KEY_A, s)
            self.assertIn(KEY_B, s)
            self.assertIn(KEY_C, s)
            self.assertNotIn("ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z", s)
        finally:
            s.unlink()

    def test_dedup_in_build(self):
        s = SharedInchiKeySet.build([KEY_A, KEY_B, KEY_A, KEY_B])
        try:
            self.assertEqual(len(s), 2)
        finally:
            s.unlink()

    def test_rejects_invalid_keys_during_build(self):
        s = SharedInchiKeySet.build([KEY_A, "too-short", "X" * 28])
        try:
            self.assertEqual(len(s), 1)
            self.assertIn(KEY_A, s)
        finally:
            s.unlink()

    def test_empty_build_raises(self):
        with self.assertRaises(ValueError):
            SharedInchiKeySet.build([])

    def test_membership_rejects_wrong_types_and_lengths(self):
        s = SharedInchiKeySet.build([KEY_A])
        try:
            self.assertNotIn("short", s)
            self.assertNotIn("X" * 28, s)
            self.assertNotIn(12345, s)
        finally:
            s.unlink()

    def test_unlink_only_by_owner(self):
        s = SharedInchiKeySet.build([KEY_A])
        try:
            attached = SharedInchiKeySet.attach(s.shm_name, len(s))
            try:
                with self.assertRaises(RuntimeError):
                    attached.unlink()
            finally:
                attached.close()
        finally:
            s.unlink()


class TestSharedInchiKeySetCrossProcess(unittest.TestCase):
    """Exercise the actual SharedMemory boundary across a spawned process."""

    def test_subprocess_can_attach_and_query(self):
        s = SharedInchiKeySet.build([KEY_A, KEY_B, KEY_C])
        try:
            ctx = mp.get_context("spawn")
            queries = [KEY_A, KEY_B, KEY_C, "ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z"]
            result_q = ctx.Queue()
            p = ctx.Process(
                target=_child_attach_and_query,
                args=(s.shm_name, len(s), queries, result_q),
            )
            p.start()
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "child process did not exit cleanly")
            results = result_q.get(timeout=5)
            self.assertEqual(results, [True, True, True, False])
        finally:
            s.unlink()
