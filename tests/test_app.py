import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import clean_text, parse_reviewer_metadata, sentiment_label  # noqa: E402


def test_metadata_parsing_keeps_review_count():
    assert parse_reviewer_metadata("1 Review") == (1.0, 0.0)
    assert parse_reviewer_metadata("96 Reviews , 1069 Followers") == (96.0, 1069.0)
    assert all(np.isnan(parse_reviewer_metadata(None)))


def test_clean_text_keeps_negations_and_separates_words():
    out = clean_text("Food was NOT good!!Service,bad http://x.com 123")
    assert "not" in out.split() and "service" in out.split() and "bad" in out.split()
    assert "http" not in out and "123" not in out


def test_sentiment_label():
    assert sentiment_label(5) == 1 and sentiment_label(1.5) == 0 and np.isnan(sentiment_label(3))


PAGES = ["Overview", "Restaurant Explorer", "Segments (Clusters)", "Find Similar",
         "Sentiment Lab", "Review Insights", "Model Report"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_sentiment_lab_predicts():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value("Sentiment Lab").run()
    at.text_area[0].set_value("Worst food ever, not fresh and very slow service").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Negative" in m.value for m in at.metric)
