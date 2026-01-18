"""
Property-Based Tests for Text Preprocessing Module

Tests the text preprocessing pipeline using hypothesis for property-based testing.
Validates Requirements 2.1 and 2.5 for Unicode normalization and output completeness.

Feature: multimodal-nlp-upgrade
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List

from text_preprocessing import (
    TextPreprocessor,
    TextPreprocessingConfig,
    PreprocessedText,
    create_preprocessor
)


# Strategy for generating Tamil Unicode text
tamil_chars = st.sampled_from(
    # Tamil Unicode range: U+0B80 to U+0BFF
    [chr(c) for c in range(0x0B80, 0x0C00) if chr(c).isprintable()]
    + list(" ")  # Include space for word separation
)

# Strategy for generating Malayalam Unicode text
malayalam_chars = st.sampled_from(
    # Malayalam Unicode range: U+0D00 to U+0D7F
    [chr(c) for c in range(0x0D00, 0x0D80) if chr(c).isprintable()]
    + list(" ")  # Include space for word separation
)

# Strategy for generating mixed Dravidian text
dravidian_text = st.text(
    alphabet=st.sampled_from(
        [chr(c) for c in range(0x0B80, 0x0C00) if chr(c).isprintable()]  # Tamil
        + [chr(c) for c in range(0x0D00, 0x0D80) if chr(c).isprintable()]  # Malayalam
        + list(" \t\n")  # Whitespace
    ),
    min_size=0,
    max_size=500
)

# Strategy for generating general Unicode text (for broader testing)
general_unicode_text = st.text(min_size=0, max_size=500)


class TestUnicodeNormalizationIdempotence:
    """
    Property 3: Unicode Normalization Idempotence
    
    *For any* text string, applying Unicode normalization twice SHALL produce
    the same result as applying it once (normalization is idempotent).
    
    **Validates: Requirements 2.1**
    """
    
    @given(text=general_unicode_text)
    @settings(max_examples=100, deadline=None)
    def test_normalization_idempotence_general(self, text: str):
        """
        Feature: multimodal-nlp-upgrade, Property 3: Unicode Normalization Idempotence
        
        For any text string, normalize(normalize(x)) == normalize(x).
        **Validates: Requirements 2.1**
        """
        preprocessor = create_preprocessor("ta", apply_morphological_segmentation=False)
        
        # Apply normalization once
        normalized_once = preprocessor.normalize(text)
        
        # Apply normalization twice
        normalized_twice = preprocessor.normalize(normalized_once)
        
        # Property: Normalization should be idempotent
        assert normalized_once == normalized_twice, (
            f"Normalization is not idempotent:\n"
            f"  Original: {repr(text)}\n"
            f"  Once: {repr(normalized_once)}\n"
            f"  Twice: {repr(normalized_twice)}"
        )
    
    @given(text=dravidian_text)
    @settings(max_examples=100, deadline=None)
    def test_normalization_idempotence_dravidian(self, text: str):
        """
        Feature: multimodal-nlp-upgrade, Property 3: Unicode Normalization Idempotence
        
        For any Dravidian text, normalize(normalize(x)) == normalize(x).
        **Validates: Requirements 2.1**
        """
        # Test with Tamil preprocessor
        preprocessor_ta = create_preprocessor("ta", apply_morphological_segmentation=False)
        
        normalized_once = preprocessor_ta.normalize(text)
        normalized_twice = preprocessor_ta.normalize(normalized_once)
        
        assert normalized_once == normalized_twice, (
            f"Tamil normalization is not idempotent:\n"
            f"  Original: {repr(text)}\n"
            f"  Once: {repr(normalized_once)}\n"
            f"  Twice: {repr(normalized_twice)}"
        )
        
        # Test with Malayalam preprocessor
        preprocessor_ml = create_preprocessor("ml", apply_morphological_segmentation=False)
        
        normalized_once_ml = preprocessor_ml.normalize(text)
        normalized_twice_ml = preprocessor_ml.normalize(normalized_once_ml)
        
        assert normalized_once_ml == normalized_twice_ml, (
            f"Malayalam normalization is not idempotent:\n"
            f"  Original: {repr(text)}\n"
            f"  Once: {repr(normalized_once_ml)}\n"
            f"  Twice: {repr(normalized_twice_ml)}"
        )
    
    @given(text=st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=('L', 'N'),  # Only letters and numbers
            blacklist_categories=('Cs',)  # Exclude surrogates
        )
    ))
    @settings(max_examples=100, deadline=None)
    def test_normalization_preserves_non_empty(self, text: str):
        """
        Feature: multimodal-nlp-upgrade, Property 3: Unicode Normalization Idempotence
        
        For any alphanumeric text, normalization should preserve content.
        **Validates: Requirements 2.1**
        """
        # Skip if text is only whitespace or empty after filtering
        assume(text.strip())
        assume(any(c.isalnum() for c in text))
        
        preprocessor = create_preprocessor("ta", apply_morphological_segmentation=False)
        
        normalized = preprocessor.normalize(text)
        
        # Property: Non-empty alphanumeric text should produce non-empty normalized text
        # Note: Some Unicode characters (like soft hyphens, zero-width chars) may be
        # legitimately removed during normalization, so we only test alphanumeric input
        assert normalized.strip(), (
            f"Non-empty alphanumeric text produced empty normalization:\n"
            f"  Original: {repr(text)}\n"
            f"  Normalized: {repr(normalized)}"
        )


class TestPreprocessingOutputCompleteness:
    """
    Property 4: Preprocessing Output Completeness
    
    *For any* input text, the Text_Preprocessor SHALL return a PreprocessedText
    object containing both normalized text and morpheme list, where neither is empty
    (for non-empty input).
    
    **Validates: Requirements 2.5**
    """
    
    @given(text=st.text(min_size=1, max_size=500, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),  # Letters, Numbers, Punctuation, Separators
        blacklist_categories=('Cs',)  # Exclude surrogates
    )))
    @settings(max_examples=100, deadline=None)
    def test_preprocessing_returns_complete_result(self, text: str):
        """
        Feature: multimodal-nlp-upgrade, Property 4: Preprocessing Output Completeness
        
        For any input text, preprocess() returns PreprocessedText with all fields.
        **Validates: Requirements 2.5**
        """
        # Skip if text is only whitespace or control characters
        assume(any(c.isalnum() for c in text))
        
        preprocessor = create_preprocessor("ta")
        
        result = preprocessor.preprocess(text)
        
        # Property: Result should be a PreprocessedText instance
        assert isinstance(result, PreprocessedText), (
            f"Result is not PreprocessedText: {type(result)}"
        )
        
        # Property: Result should have original text
        assert result.original == text, (
            f"Original text not preserved:\n"
            f"  Input: {repr(text)}\n"
            f"  Result.original: {repr(result.original)}"
        )
        
        # Property: Result should have normalized text (non-empty for non-empty input)
        assert result.normalized is not None, "Normalized text is None"
        
        # Property: Result should have morphemes list
        assert result.morphemes is not None, "Morphemes list is None"
        assert isinstance(result.morphemes, list), "Morphemes is not a list"
        
        # Property: Result should have morpheme_text
        assert result.morpheme_text is not None, "Morpheme text is None"
        
        # Property: For non-empty alphanumeric input, should have non-empty output
        if any(c.isalnum() for c in text):
            assert result.normalized.strip() or result.morphemes, (
                f"Non-empty input produced empty output:\n"
                f"  Input: {repr(text)}\n"
                f"  Normalized: {repr(result.normalized)}\n"
                f"  Morphemes: {result.morphemes}"
            )
    
    @given(text=dravidian_text)
    @settings(max_examples=100, deadline=None)
    def test_preprocessing_dravidian_completeness(self, text: str):
        """
        Feature: multimodal-nlp-upgrade, Property 4: Preprocessing Output Completeness
        
        For any Dravidian text, preprocess() returns complete PreprocessedText.
        **Validates: Requirements 2.5**
        """
        # Skip empty text
        assume(text.strip())
        
        preprocessor = create_preprocessor("ta")
        
        result = preprocessor.preprocess(text)
        
        # Property: All fields should be present
        assert hasattr(result, 'original'), "Missing 'original' field"
        assert hasattr(result, 'normalized'), "Missing 'normalized' field"
        assert hasattr(result, 'morphemes'), "Missing 'morphemes' field"
        assert hasattr(result, 'morpheme_text'), "Missing 'morpheme_text' field"
        assert hasattr(result, 'language'), "Missing 'language' field"
        
        # Property: Morpheme text should be space-joined morphemes
        if result.morphemes:
            expected_morpheme_text = " ".join(result.morphemes)
            assert result.morpheme_text == expected_morpheme_text, (
                f"Morpheme text mismatch:\n"
                f"  Expected: {repr(expected_morpheme_text)}\n"
                f"  Got: {repr(result.morpheme_text)}"
            )
    
    @given(text=st.text(min_size=0, max_size=100))
    @settings(max_examples=100, deadline=None)
    def test_preprocessing_handles_empty_input(self, text: str):
        """
        Feature: multimodal-nlp-upgrade, Property 4: Preprocessing Output Completeness
        
        For any input (including empty), preprocess() should not raise exceptions.
        **Validates: Requirements 2.5**
        """
        preprocessor = create_preprocessor("ta")
        
        # Should not raise any exceptions
        try:
            result = preprocessor.preprocess(text)
            
            # Result should always be a PreprocessedText
            assert isinstance(result, PreprocessedText)
            
            # For empty input, should return empty but valid result
            if not text:
                assert result.original == ""
                assert result.normalized == ""
                assert result.morphemes == []
                assert result.morpheme_text == ""
                
        except Exception as e:
            pytest.fail(f"preprocess() raised exception for input {repr(text)}: {e}")
    
    @given(
        text=st.text(min_size=1, max_size=200),
        language=st.sampled_from(["ta", "ml", "tamil", "malayalam"])
    )
    @settings(max_examples=100, deadline=None)
    def test_preprocessing_language_field(self, text: str, language: str):
        """
        Feature: multimodal-nlp-upgrade, Property 4: Preprocessing Output Completeness
        
        For any input and language, result should have correct language field.
        **Validates: Requirements 2.5**
        """
        preprocessor = create_preprocessor(language)
        
        result = preprocessor.preprocess(text)
        
        # Property: Language should be normalized to ISO code
        expected_lang = "ta" if language in ["ta", "tamil"] else "ml"
        assert result.language == expected_lang, (
            f"Language mismatch: expected {expected_lang}, got {result.language}"
        )


class TestPreprocessedTextSerialization:
    """Tests for PreprocessedText serialization round-trip."""
    
    @given(
        original=st.text(min_size=0, max_size=200),
        normalized=st.text(min_size=0, max_size=200),
        morphemes=st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=20),
        language=st.sampled_from(["ta", "ml"])
    )
    @settings(max_examples=100, deadline=None)
    def test_preprocessed_text_round_trip(
        self,
        original: str,
        normalized: str,
        morphemes: List[str],
        language: str
    ):
        """
        PreprocessedText should survive serialization round-trip.
        """
        morpheme_text = " ".join(morphemes)
        
        original_obj = PreprocessedText(
            original=original,
            normalized=normalized,
            morphemes=morphemes,
            morpheme_text=morpheme_text,
            language=language
        )
        
        # Serialize to dict
        data = original_obj.to_dict()
        
        # Deserialize back
        restored = PreprocessedText.from_dict(data)
        
        # Properties should be preserved
        assert restored.original == original_obj.original
        assert restored.normalized == original_obj.normalized
        assert restored.morphemes == original_obj.morphemes
        assert restored.morpheme_text == original_obj.morpheme_text
        assert restored.language == original_obj.language


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
