"""Logging configuration for the application."""

import logging
import os


def setup_logging(level: int | str | None = None) -> None:
	"""Configure application-wide logging.

	Uses the provided level, the ``LOG_LEVEL`` environment variable, or INFO.
	"""
	configured_level = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
	if isinstance(configured_level, str):
		configured_level = getattr(logging, configured_level.upper(), logging.INFO)

	logging.basicConfig(
		level=configured_level,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	logging.getLogger().setLevel(configured_level)
