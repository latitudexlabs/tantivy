mod fuzzy_phrase_query;
mod fuzzy_phrase_weight;
mod fuzzy_phrase_scorer;

pub use self::fuzzy_phrase_query::FuzzyPhraseQuery;
pub(crate) use self::fuzzy_phrase_weight::FuzzyPhraseWeight;
pub(crate) use self::fuzzy_phrase_scorer::FuzzyPhraseScorer;
