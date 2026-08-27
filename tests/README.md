# Tests

CPU-safe contract tests and optional GPU/vendor parity tests live here. Run
the full discovery pass from the repository root with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests that need the pinned Atari environment skip themselves when the required
dependencies are unavailable.
