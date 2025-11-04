"""Utility helpers for Selenium-based harvesting."""

from __future__ import annotations

from typing import Iterable, Tuple

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException


Selector = Tuple[str, str]


def resilient_find_element(driver: WebDriver, selectors: Iterable[Selector], name: str) -> WebElement:
    """Locate an element using a sequence of selector fallbacks.

    Each selector should be a tuple of (By, value). The function attempts them
    in order and returns the first successfully located element. If all
    selectors fail, a descriptive ``NoSuchElementException`` is raised to make
    downstream debugging easier.
    """

    last_error: Exception | None = None

    for by, value in selectors:
        try:
            return driver.find_element(by, value)
        except NoSuchElementException as exc:
            last_error = exc
            continue
        except Exception as exc:  # Catch intermittent driver issues without aborting the sequence.
            last_error = exc
            continue

    message = f"All selector fallbacks exhausted for '{name}'."
    if last_error is not None:
        raise NoSuchElementException(message) from last_error
    raise NoSuchElementException(message)

