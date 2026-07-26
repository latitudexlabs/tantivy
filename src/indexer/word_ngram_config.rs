use serde::{Deserialize, Serialize};

/// Configuration for word-level ngram indexing.
///
/// This enables frequency-based word ngram indexing where ngrams are created
/// based on whether individual terms are frequent or rare in the corpus.
/// This can significantly speed up phrase queries at the cost of increased index size.
///
/// Frequent terms are determined by analyzing the corpus during indexing.
/// A term is considered "frequent" if it appears in more than a certain percentage
/// of documents (configurable via `frequent_term_threshold`).
///
/// # Example
///
/// ```rust
/// use tantivy::indexer::WordNgramSet;
///
/// // Enable all bigram types
/// let config = WordNgramSet::new()
///     .with_ngram_ff()
///     .with_ngram_fr()
///     .with_ngram_rf();
///
/// // Enable only frequent-frequent bigrams and trigrams
/// let config = WordNgramSet::new()
///     .with_ngram_ff()
///     .with_ngram_fff();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordNgramSet {
    bits: u8,
}

impl WordNgramSet {
    /// No ngrams - only single terms are indexed (default)
    pub const NONE: u8 = 0b00000000;
    /// Bigram: Frequent-Frequent
    /// Index bigrams where both words are frequent (e.g., "the world")
    pub const NGRAM_FF: u8 = 0b00000001;
    /// Bigram: Frequent-Rare
    /// Index bigrams where first word is frequent, second is rare (e.g., "the elephant")
    pub const NGRAM_FR: u8 = 0b00000010;
    /// Bigram: Rare-Frequent
    /// Index bigrams where first word is rare, second is frequent (e.g., "elephant the")
    pub const NGRAM_RF: u8 = 0b00000100;
    /// Trigram: Frequent-Frequent-Frequent
    /// Index trigrams where all three words are frequent (e.g., "in the world")
    pub const NGRAM_FFF: u8 = 0b00001000;
    /// Trigram: Rare-Frequent-Frequent
    /// Index trigrams where first is rare, others are frequent
    pub const NGRAM_RFF: u8 = 0b00010000;
    /// Trigram: Frequent-Frequent-Rare
    /// Index trigrams where last is rare, others are frequent
    pub const NGRAM_FFR: u8 = 0b00100000;
    /// Trigram: Frequent-Rare-Frequent
    /// Index trigrams where middle is rare, others are frequent
    pub const NGRAM_FRF: u8 = 0b01000000;

    /// Create a new empty ngram set
    pub const fn new() -> Self {
        WordNgramSet { bits: Self::NONE }
    }

    /// Create from bits value
    pub const fn from_bits(bits: u8) -> Self {
        WordNgramSet { bits }
    }

    /// Get the raw bits
    pub const fn bits(&self) -> u8 {
        self.bits
    }

    /// Check if a flag is set
    pub const fn contains(&self, flag: u8) -> bool {
        (self.bits & flag) != 0
    }

    /// Add a flag
    pub const fn with(self, flag: u8) -> Self {
        WordNgramSet {
            bits: self.bits | flag,
        }
    }

    /// Check if empty
    pub const fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Builder method for NGRAM_FF
    pub const fn with_ngram_ff(self) -> Self {
        self.with(Self::NGRAM_FF)
    }

    /// Builder method for NGRAM_FR
    pub const fn with_ngram_fr(self) -> Self {
        self.with(Self::NGRAM_FR)
    }

    /// Builder method for NGRAM_RF
    pub const fn with_ngram_rf(self) -> Self {
        self.with(Self::NGRAM_RF)
    }

    /// Builder method for NGRAM_FFF
    pub const fn with_ngram_fff(self) -> Self {
        self.with(Self::NGRAM_FFF)
    }

    /// Builder method for NGRAM_RFF
    pub const fn with_ngram_rff(self) -> Self {
        self.with(Self::NGRAM_RFF)
    }

    /// Builder method for NGRAM_FFR
    pub const fn with_ngram_ffr(self) -> Self {
        self.with(Self::NGRAM_FFR)
    }

    /// Builder method for NGRAM_FRF
    pub const fn with_ngram_frf(self) -> Self {
        self.with(Self::NGRAM_FRF)
    }
}

impl Default for WordNgramSet {
    fn default() -> Self {
        WordNgramSet::new()
    }
}

impl std::ops::BitOr for WordNgramSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        WordNgramSet {
            bits: self.bits | rhs.bits,
        }
    }
}

/// Configuration for word-level ngram indexing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordNgramConfig {
    /// Which ngram patterns to index
    pub ngram_set: WordNgramSet,
    
    /// Threshold for determining if a term is "frequent"
    /// A term is frequent if it appears in more than this fraction of documents
    /// Default: 0.01 (1% of documents)
    #[serde(default = "default_frequent_threshold")]
    pub frequent_term_threshold: f32,
    
    /// Maximum number of frequent terms to track
    /// Default: 10,000
    #[serde(default = "default_max_frequent_terms")]
    pub max_frequent_terms: usize,
    
    /// Generate all ordered combinations instead of just consecutive ngrams
    /// 
    /// When false (default): generates consecutive ngrams only
    /// - For "a b c d": generates ab, bc, cd (bigrams) and abc, bcd, cde (trigrams)
    /// 
    /// When true: generates all ordered combinations within a sliding window
    /// - For "a b c d e" with window size 5: generates combinations like ab, ac, ad, ae, bc, bd, be, etc.
    /// 
    /// This mode enables better fuzzy phrase query optimization by matching documents 
    /// with gaps between terms, at the cost of increased index size.
    /// 
    /// **Performance:** Optimized to only generate combinations with the newest token
    /// in the sliding window, reducing computational overhead significantly.
    /// 
    /// **Window size:** Controlled by `all_combinations_window_size` (default: 100)
    /// 
    /// Default: false
    #[serde(default)]
    pub all_combinations: bool,
    
    /// Maximum window size for all_combinations mode
    /// 
    /// Controls how many tokens to keep in the sliding window when generating
    /// all ordered combinations. Smaller values reduce index size and indexing time,
    /// but may miss combinations with larger gaps.
    /// 
    /// For a window size of N:
    /// - Bigrams per token: N - 1
    /// - Trigrams per token: (N-1)*(N-2)/2
    /// 
    /// Examples:
    /// - N=3: 2 bigrams + 1 trigram = 3 combinations per token
    /// - N=5: 4 bigrams + 6 trigrams = 10 combinations per token
    /// - N=10: 9 bigrams + 36 trigrams = 45 combinations per token
    /// 
    /// Default: 100
    #[serde(default = "default_all_combinations_window_size")]
    pub all_combinations_window_size: usize,
    
    /// Enable edge ngram generation for each token before creating combinations
    /// 
    /// When enabled, each token is first split into progressive prefixes (edge ngrams)
    /// starting from min_edge_ngram length, then bigrams/trigrams are created from
    /// all combinations of these edge ngrams.
    /// 
    /// Example with "test1 xyz" and min_edge_ngram=2:
    /// - Edge ngrams: test1, test, tes, te, xyz, xy
    /// - Bigrams: test1 xyz, test1 xy, test xyz, test xy, tes xyz, tes xy, te xyz, te xy
    /// 
    /// This enables prefix-based matching in phrase queries, useful for autocomplete
    /// and partial match scenarios.
    /// 
    /// Default: false
    #[serde(default)]
    pub edge_ngram: bool,
    
    /// Minimum length for edge ngrams
    /// 
    /// When edge_ngram is enabled, this controls the minimum length of edge ngrams
    /// to generate. Edge ngrams are generated from this length up to the full token length.
    /// 
    /// For example, with min_edge_ngram=2:
    /// - "hello" -> hello, hell, hel, he (stops at 2 chars)
    /// - "hi" -> hi (only full token, since it's already at minimum)
    /// - "a" -> a (only full token, below minimum)
    /// 
    /// Smaller values generate more edge ngrams, increasing index size but enabling
    /// shorter prefix matches.
    /// 
    /// Default: 2
    #[serde(default = "default_min_edge_ngram")]
    pub min_edge_ngram: usize,

    /// Also index each token's edge ngrams as standalone (unigram) terms
    /// 
    /// `edge_ngram` alone only embeds prefixes inside bigram/trigram
    /// combination terms, so a single-word prefix ("berw") has no term to
    /// match. With this enabled, every prefix of a token from
    /// `min_edge_ngram` up to its full length (exclusive) is additionally
    /// indexed as its own term **at the token's position**, so both plain
    /// term lookups and position-based phrase matching work on partially
    /// typed words.
    /// 
    /// Requires `edge_ngram` to be enabled; shares `min_edge_ngram`.
    /// 
    /// Default: false
    #[serde(default)]
    pub unigram_edge_ngram: bool,
}

fn default_frequent_threshold() -> f32 {
    0.01
}

fn default_max_frequent_terms() -> usize {
    10_000
}

fn default_all_combinations_window_size() -> usize {
    100
}

fn default_min_edge_ngram() -> usize {
    2
}

impl Default for WordNgramConfig {
    fn default() -> Self {
        WordNgramConfig {
            ngram_set: WordNgramSet::new(),
            frequent_term_threshold: default_frequent_threshold(),
            max_frequent_terms: default_max_frequent_terms(),
            all_combinations: false,
            all_combinations_window_size: default_all_combinations_window_size(),
            edge_ngram: false,
            min_edge_ngram: default_min_edge_ngram(),
            unigram_edge_ngram: false,
        }
    }
}

impl WordNgramConfig {
    /// Create a new word ngram configuration with specific flags
    pub fn new(bits: u8) -> Self {
        WordNgramConfig {
            ngram_set: WordNgramSet::from_bits(bits),
            ..Default::default()
        }
    }

    /// Create a new word ngram configuration with a WordNgramSet
    pub fn with_set(ngram_set: WordNgramSet) -> Self {
        WordNgramConfig {
            ngram_set,
            ..Default::default()
        }
    }
    
    /// Set the frequent term threshold
    pub fn with_frequent_threshold(mut self, threshold: f32) -> Self {
        self.frequent_term_threshold = threshold;
        self
    }
    
    /// Set the maximum number of frequent terms to track
    pub fn with_max_frequent_terms(mut self, max: usize) -> Self {
        self.max_frequent_terms = max;
        self
    }
    
    /// Enable all ordered combinations mode
    /// 
    /// When enabled, generates all ordered combinations of terms within a sliding window
    /// instead of just consecutive pairs/triplets. This enables better fuzzy phrase 
    /// query optimization but increases index size.
    /// 
    /// Uses default window size of 5 tokens. Use `with_all_combinations_window_size()`
    /// to customize the window size.
    pub fn with_all_combinations(mut self) -> Self {
        self.all_combinations = true;
        self
    }
    
    /// Enable edge ngram generation
    /// 
    /// When enabled, each token is split into progressive prefixes before creating
    /// ngram combinations. Uses default minimum length of 2.
    pub fn with_edge_ngram(mut self) -> Self {
        self.edge_ngram = true;
        self
    }
    
    /// Set the minimum length for edge ngrams
    /// 
    /// Controls the minimum length of edge ngrams to generate. Edge ngrams are
    /// generated from this length up to the full token length.
    pub fn with_min_edge_ngram(mut self, min_length: usize) -> Self {
        self.min_edge_ngram = min_length;
        self
    }

    /// Also index each token's edge ngrams as standalone terms at the
    /// token's position (see [`WordNgramConfig::unigram_edge_ngram`]).
    pub fn with_unigram_edge_ngram(mut self) -> Self {
        self.unigram_edge_ngram = true;
        self
    }
    
    /// Set the window size for all_combinations mode
    /// 
    /// Controls how many tokens to keep in the sliding window when generating
    /// all ordered combinations. Smaller values reduce index size, larger values
    /// capture combinations with bigger gaps.
    /// 
    /// Recommended range: 3-7 tokens
    /// - 3: Minimal memory, only short-range combinations
    /// - 5: Good balance (default)
    /// - 7+: More coverage but exponential growth in ngrams
    pub fn with_all_combinations_window_size(mut self, size: usize) -> Self {
        self.all_combinations_window_size = size;
        self
    }
    
    /// Returns true if any ngrams are enabled
    pub fn is_enabled(&self) -> bool {
        !self.ngram_set.is_empty()
    }
    
    /// Check if this configuration includes any bigrams
    pub fn contains_bigrams(&self) -> bool {
        self.ngram_set.contains(WordNgramSet::NGRAM_FF)
            || self.ngram_set.contains(WordNgramSet::NGRAM_FR)
            || self.ngram_set.contains(WordNgramSet::NGRAM_RF)
    }
    
    /// Check if this configuration includes any trigrams
    pub fn contains_trigrams(&self) -> bool {
        self.ngram_set.contains(WordNgramSet::NGRAM_FFF)
            || self.ngram_set.contains(WordNgramSet::NGRAM_RFF)
            || self.ngram_set.contains(WordNgramSet::NGRAM_FFR)
            || self.ngram_set.contains(WordNgramSet::NGRAM_FRF)
    }
    
    /// Check if a specific ngram type is enabled
    pub fn has_ngram_type(&self, ngram_type: &NgramType) -> bool {
        match ngram_type {
            NgramType::SingleTerm => true, // Always enabled
            NgramType::NgramFF => self.ngram_set.contains(WordNgramSet::NGRAM_FF),
            NgramType::NgramFR => self.ngram_set.contains(WordNgramSet::NGRAM_FR),
            NgramType::NgramRF => self.ngram_set.contains(WordNgramSet::NGRAM_RF),
            NgramType::NgramFFF => self.ngram_set.contains(WordNgramSet::NGRAM_FFF),
            NgramType::NgramRFF => self.ngram_set.contains(WordNgramSet::NGRAM_RFF),
            NgramType::NgramFFR => self.ngram_set.contains(WordNgramSet::NGRAM_FFR),
            NgramType::NgramFRF => self.ngram_set.contains(WordNgramSet::NGRAM_FRF),
        }
    }
    
    /// Check if all bigram types are enabled
    /// When true, frequency tracking is not needed for bigram classification
    pub fn has_all_bigram_types(&self) -> bool {
        self.ngram_set.contains(WordNgramSet::NGRAM_FF)
            && self.ngram_set.contains(WordNgramSet::NGRAM_FR)
            && self.ngram_set.contains(WordNgramSet::NGRAM_RF)
    }
    
    /// Check if all trigram types are enabled
    /// When true, frequency tracking is not needed for trigram classification
    pub fn has_all_trigram_types(&self) -> bool {
        self.ngram_set.contains(WordNgramSet::NGRAM_FFF)
            && self.ngram_set.contains(WordNgramSet::NGRAM_RFF)
            && self.ngram_set.contains(WordNgramSet::NGRAM_FFR)
            && self.ngram_set.contains(WordNgramSet::NGRAM_FRF)
    }
    
    /// Create a builder for WordNgramConfig
    pub fn builder() -> WordNgramConfigBuilder {
        WordNgramConfigBuilder::default()
    }
}

/// Builder for WordNgramConfig
#[derive(Default)]
pub struct WordNgramConfigBuilder {
    ngram_set: WordNgramSet,
    frequent_term_threshold: Option<f32>,
    max_frequent_terms: Option<usize>,
    all_combinations: bool,
    all_combinations_window_size: Option<usize>,
    edge_ngram: bool,
    min_edge_ngram: Option<usize>,
    unigram_edge_ngram: bool,
}

impl WordNgramConfigBuilder {
    /// Set the ngram types to enable
    pub fn ngram_types(mut self, ngram_set: WordNgramSet) -> Self {
        self.ngram_set = ngram_set;
        self
    }
    
    /// Set the frequent term threshold
    pub fn frequent_threshold(mut self, threshold: f32) -> Self {
        self.frequent_term_threshold = Some(threshold);
        self
    }
    
    /// Set the maximum number of frequent terms
    pub fn max_frequent_terms(mut self, max: usize) -> Self {
        self.max_frequent_terms = Some(max);
        self
    }
    
    /// Enable all ordered combinations mode
    pub fn all_combinations(mut self, enabled: bool) -> Self {
        self.all_combinations = enabled;
        self
    }
    
    /// Set the window size for all_combinations mode
    pub fn all_combinations_window_size(mut self, size: usize) -> Self {
        self.all_combinations_window_size = Some(size);
        self
    }
    
    /// Enable edge ngram generation
    pub fn edge_ngram(mut self, enabled: bool) -> Self {
        self.edge_ngram = enabled;
        self
    }
    
    /// Set the minimum length for edge ngrams
    pub fn min_edge_ngram(mut self, min_length: usize) -> Self {
        self.min_edge_ngram = Some(min_length);
        self
    }
    
    /// Also index each token's edge ngrams as standalone terms
    pub fn unigram_edge_ngram(mut self, enabled: bool) -> Self {
        self.unigram_edge_ngram = enabled;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> WordNgramConfig {
        WordNgramConfig {
            ngram_set: self.ngram_set,
            frequent_term_threshold: self.frequent_term_threshold.unwrap_or_else(default_frequent_threshold),
            max_frequent_terms: self.max_frequent_terms.unwrap_or_else(default_max_frequent_terms),
            all_combinations: self.all_combinations,
            all_combinations_window_size: self.all_combinations_window_size.unwrap_or_else(default_all_combinations_window_size),
            edge_ngram: self.edge_ngram,
            min_edge_ngram: self.min_edge_ngram.unwrap_or_else(default_min_edge_ngram),
            unigram_edge_ngram: self.unigram_edge_ngram,
        }
    }
}

/// Ngram type identifier for classifying bigrams and trigrams based on term frequency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum NgramType {
    /// Single term (not an ngram)
    #[default]
    SingleTerm = 0,
    /// Bigram: Frequent-Frequent
    NgramFF = 1,
    /// Bigram: Frequent-Rare
    NgramFR = 2,
    /// Bigram: Rare-Frequent
    NgramRF = 3,
    /// Trigram: Frequent-Frequent-Frequent
    NgramFFF = 4,
    /// Trigram: Rare-Frequent-Frequent
    NgramRFF = 5,
    /// Trigram: Frequent-Frequent-Rare
    NgramFFR = 6,
    /// Trigram: Frequent-Rare-Frequent
    NgramFRF = 7,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_set_flags() {
        let config = WordNgramSet::new()
            .with_ngram_ff()
            .with_ngram_fr();
        assert!(config.contains(WordNgramSet::NGRAM_FF));
        assert!(config.contains(WordNgramSet::NGRAM_FR));
        assert!(!config.contains(WordNgramSet::NGRAM_RF));
    }

    #[test]
    fn test_edge_ngram_config_defaults() {
        let config = WordNgramConfig::default();
        assert!(!config.edge_ngram, "edge_ngram should be disabled by default");
        assert_eq!(config.min_edge_ngram, 2, "min_edge_ngram should default to 2");
    }

    #[test]
    fn test_edge_ngram_config_builder() {
        let config = WordNgramConfig::builder()
            .ngram_types(WordNgramSet::new().with_ngram_ff())
            .edge_ngram(true)
            .min_edge_ngram(3)
            .build();

        assert!(config.edge_ngram);
        assert_eq!(config.min_edge_ngram, 3);
        assert!(config.ngram_set.contains(WordNgramSet::NGRAM_FF));
    }

    #[test]
    fn test_edge_ngram_config_fluent() {
        let config = WordNgramConfig::with_set(WordNgramSet::new().with_ngram_ff())
            .with_edge_ngram()
            .with_min_edge_ngram(4);

        assert!(config.edge_ngram);
        assert_eq!(config.min_edge_ngram, 4);
    }

    #[test]
    fn test_edge_ngram_with_all_combinations() {
        let config = WordNgramConfig::builder()
            .ngram_types(WordNgramSet::new().with_ngram_ff().with_ngram_fr())
            .all_combinations(true)
            .all_combinations_window_size(7)
            .edge_ngram(true)
            .min_edge_ngram(2)
            .build();

        assert!(config.edge_ngram);
        assert_eq!(config.min_edge_ngram, 2);
        assert!(config.all_combinations);
        assert_eq!(config.all_combinations_window_size, 7);
    }

    #[test]
    fn test_unigram_edge_ngram_deserializes_absent_as_false() {
        // Configs serialized before the field existed must load unchanged.
        let json = r#"{"ngram_set":{"bits":1},"all_combinations":true,"edge_ngram":true,"min_edge_ngram":3}"#;
        let config: WordNgramConfig = serde_json::from_str(json).unwrap();
        assert!(!config.unigram_edge_ngram);
        assert!(config.edge_ngram);

        let config = WordNgramConfig::with_set(WordNgramSet::new().with_ngram_ff())
            .with_unigram_edge_ngram();
        let round: WordNgramConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert!(round.unigram_edge_ngram);
    }

    #[test]
    fn test_edge_ngram_serialization() {
        let config = WordNgramConfig::builder()
            .ngram_types(WordNgramSet::new().with_ngram_ff())
            .edge_ngram(true)
            .min_edge_ngram(3)
            .build();

        // Test that it can be serialized and deserialized
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WordNgramConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.edge_ngram, deserialized.edge_ngram);
        assert_eq!(config.min_edge_ngram, deserialized.min_edge_ngram);
        assert_eq!(config.ngram_set, deserialized.ngram_set);
    }
}
