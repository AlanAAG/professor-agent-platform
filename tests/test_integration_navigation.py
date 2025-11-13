import re
from typing import List

import pytest
from lxml import html
from lxml.cssselect import CSSSelector

from src.harvester.scraping import (
    process_resource_links,
    _extract_title_robust,
    _extract_date_robust,
    classify_url,
)


class FakeElement:
    """A minimal Selenium WebElement-like wrapper backed by lxml for testing."""

    def __init__(self, node):
        self._node = node

    @property
    def text(self):
        return self._node.text_content()

    def get_attribute(self, name: str):
        # lxml uses .get for attributes
        return self._node.get(name)

    def find_element(self, by, selector: str):
        if by == 'xpath' or by.__class__.__name__ == 'By' and str(by).endswith('XPATH'):
            res = self._node.xpath(selector)
            if not res:
                raise Exception('No element')
            return FakeElement(res[0])
        if by == 'css selector' or by.__class__.__name__ == 'By' and str(by).endswith('CSS_SELECTOR'):
            sel = CSSSelector(selector)
            res = sel(self._node)
            if not res:
                raise Exception('No element')
            return FakeElement(res[0])
        raise NotImplementedError(by)

    def find_elements(self, by, selector: str) -> List['FakeElement']:
        if by == 'xpath' or by.__class__.__name__ == 'By' and str(by).endswith('XPATH'):
            return [FakeElement(n) for n in self._node.xpath(selector)]
        if by == 'css selector' or by.__class__.__name__ == 'By' and str(by).endswith('CSS_SELECTOR'):
            sel = CSSSelector(selector)
            return [FakeElement(n) for n in sel(self._node)]
        return []


def _parse_fragment(html_str: str) -> FakeElement:
    node = html.fragment_fromstring(html_str, create_parent=True)
    return FakeElement(node)


def _first_anchor(root: FakeElement) -> FakeElement:
    return root.find_element('xpath', './/a')


# --- Unit test examples for helper functions ---

def test_title_from_anchor_text():
    frag = _parse_fragment('<a href="/docs"><p>Openai</p></a>')
    a = _first_anchor(frag)
    title = _extract_title_robust(a, 'https://example.com/docs')
    assert title == 'Openai'


def test_title_from_filecontentcol_p():
    html_str = '''
    <div class="fileBox"><div class="fileBoxRow">
      <div class="fileContentCol"><a href="/p"><p>Session 2 PPT</p></a></div>
    </div></div>
    '''
    root = _parse_fragment(html_str)
    a = _first_anchor(root)
    title = _extract_title_robust(a, 'https://host/p')
    assert 'Session 2 PPT' in title


def test_title_from_fileboxend():
    html_str = '''
    <div class="fileBox"><div class="fileBoxRow">
      <div class="fileContentCol"><a href="/p"> <span></span> </a></div>
    </div></div>
    <div class="fileBoxend"><a>How does a machine learn to speak our language?</a><span>29/09/2025</span></div>
    '''
    root = _parse_fragment(html_str)
    a = _first_anchor(root)
    title = _extract_title_robust(a, 'https://h/p')
    assert title.startswith('How does a machine')


def test_date_extraction_numeric():
    html_str = '<div class="fileBoxend"><span>29/09/2025</span></div>'
    root = _parse_fragment(html_str)
    date = _extract_date_robust(root)
    assert date == '29/09/2025'


def test_date_extraction_textual_month():
    html_str = '<div class="fileBox"><div class="fileContentCol"><span>September 29, 2025</span></div></div>'
    root = _parse_fragment(html_str)
    date = _extract_date_robust(root)
    assert re.search(r'29|September', date)


# --- Integration-like tests for process_resource_links ---

def test_pattern1_standard_resource_link():
    html_str = '''
    <div class="fileBox">
        <div class="fileBoxRow">
            <div class="fileContentCol">
                <a href="https://platform.openai.com/docs"><p>Openai</p></a>
            </div>
        </div>
    </div>
    <div class="fileBoxend">
        <a>How does a machine learn to speak our language?</a>
        <span>29/09/2025</span>
    </div>
    '''
    root = _parse_fragment(html_str)
    links = [e for e in root.find_elements('xpath', './/a')]
    out = process_resource_links(None, links)
    item = out[0]
    assert item['url'].startswith('https://platform.openai.com')
    assert 'Openai' in item['title'] or 'How does a machine' in item['title']
    assert item['date'] == '29/09/2025'


def test_pattern2_session_recording_primary_is_icon_link():
    html_str = '''
    <div class="fileBox">
        <div class="fileBoxRow">
            <img src="recording-icon.svg">
            <div class="fileContentCol">
                <p>How does a machine learn to speak our language?</p>
                <span>29/09/2025</span>
            </div>
        </div>
        <a href="https://drive.google.com/file/d/abc/view"><svg></svg></a>
    </div>
    '''
    root = _parse_fragment(html_str)
    # Pass container to allow robust URL search
    out = process_resource_links(None, [root])
    assert out[0]['url'].startswith('https://drive.google.com/file')
    assert out[0]['type'] in ('DRIVE_RECORDING', 'RECORDING_LINK')


def test_pattern3_in_class_material_relative_urls_resolved(monkeypatch):
    # Simulate BASE_URL
    import src.harvester.config as cfg
    monkeypatch.setattr(cfg, 'BASE_URL', 'https://site.example/')
    html_str = '''
    <div class="fileBox"><div class="fileBoxRow">
        <div class="fileContentCol">
            <a href="/presentation/123"><p>Session 2 PPT</p></a>
        </div>
    </div></div>
    <div class="fileBoxend"><span>30/09/2025</span></div>
    '''
    root = _parse_fragment(html_str)
    out = process_resource_links(None, [root])
    assert out[0]['url'] == 'https://site.example/presentation/123'
    assert out[0]['date'] == '30/09/2025'


def test_classify_url_variants():
    assert classify_url('https://docs.google.com/document/d/1/edit') == 'GOOGLE_DOCS'
    assert classify_url('https://docs.google.com/spreadsheets/d/1/edit') == 'GOOGLE_SHEETS'
    assert classify_url('https://docs.google.com/presentation/d/1/edit') == 'GOOGLE_SLIDES'
    assert classify_url('https://view.officeapps.live.com/op/view.aspx?src=...') == 'OFFICE_ONLINE_VIEWER'
    assert classify_url('https://host/file.pptx') == 'OFFICE_DOCUMENT'
    assert classify_url('https://zoom.us/rec/abc') == 'RECORDING_ZOOM'
    assert classify_url('https://drive.google.com/file/d/abc/view') == 'DRIVE_RECORDING'
    assert classify_url('https://www.youtube.com/watch?v=1') == 'YOUTUBE_VIDEO'
    assert classify_url('https://example.com/file.pdf') == 'PDF_DOCUMENT'


def test_selector_robust_to_classname_variation():
    # Change class to different casing; selectors should be case-insensitive where relevant
    html_str = '''
    <div class="filebox">
      <div class="fileboxrow">
        <div class="fileContentCOL"><a href="/r"><p>Title Here</p></a></div>
      </div>
    </div>
    <div class="FILEBOXEND"><span>September 29, 2025</span></div>
    '''
    root = _parse_fragment(html_str)
    out = process_resource_links(None, [root])
    assert out and re.search(r'\d{4}', out[0]['date'])
    assert 'Title Here' in out[0]['title']
