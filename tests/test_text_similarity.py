from agent_controller.text_similarity import fuzzy_overlap_count, tokenize


def test_tokenize_lowercases_and_filters_stopwords_and_short_words():
    words = tokenize("The Quick Brown fox is a test")
    assert "quick" in words
    assert "brown" in words
    assert "the" not in words  # stopword
    assert "is" not in words  # stopword
    assert "a" not in words  # too short


def test_fuzzy_overlap_exact_match():
    assert fuzzy_overlap_count({"function"}, {"function"}) == 1


def test_fuzzy_overlap_prefix_match():
    assert fuzzy_overlap_count({"called"}, {"callers"}) == 1


def test_fuzzy_overlap_no_match_for_unrelated_words():
    assert fuzzy_overlap_count({"function"}, {"directory"}) == 0


def test_fuzzy_overlap_counts_each_query_word_once():
    assert fuzzy_overlap_count({"called", "function"}, {"callers"}) == 1
