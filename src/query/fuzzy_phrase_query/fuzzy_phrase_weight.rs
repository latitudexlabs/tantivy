use super::FuzzyPhraseScorer;
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::{Explanation, Scorer, Weight};
use crate::schema::{IndexRecordOption, Term};
use crate::{DocId, DocSet, Score};
use std::sync::Arc;

pub struct FuzzyPhraseWeight {
    phrase_terms: Arc<Vec<Term>>,
    min_match: usize,
    similarity_weight_opt: Option<Bm25Weight>,
}

impl FuzzyPhraseWeight {
    /// Creates a new fuzzy phrase weight.
    /// If `similarity_weight_opt` is None, then scoring is disabled
    pub fn new(
        phrase_terms: Arc<Vec<Term>>,
        min_match: usize,
        similarity_weight_opt: Option<Bm25Weight>,
    ) -> FuzzyPhraseWeight {
        FuzzyPhraseWeight {
            phrase_terms,
            min_match,
            similarity_weight_opt,
        }
    }

    fn fieldnorm_reader(&self, reader: &SegmentReader) -> crate::Result<FieldNormReader> {
        let field = self.phrase_terms[0].field();
        if self.similarity_weight_opt.is_some() {
            if let Some(fieldnorm_reader) = reader.fieldnorms_readers().get_field(field)? {
                return Ok(fieldnorm_reader);
            }
        }
        Ok(FieldNormReader::constant(reader.max_doc(), 1))
    }

    pub(crate) fn fuzzy_phrase_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<FuzzyPhraseScorer<SegmentPostings>> {
        let similarity_weight_opt = self
            .similarity_weight_opt
            .as_ref()
            .map(|similarity_weight| similarity_weight.boost_by(boost));
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;

        // The scorer drives candidate generation and position verification
        // from a single set of postings.
        let mut term_postings_list = Vec::with_capacity(self.phrase_terms.len());
        for term in self.phrase_terms.iter() {
            let postings = reader
                .inverted_index(term.field())?
                .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
                // Term doesn't exist in this segment.
                .unwrap_or_else(SegmentPostings::empty);
            term_postings_list.push(postings);
        }

        Ok(FuzzyPhraseScorer::new(
            term_postings_list,
            self.min_match,
            similarity_weight_opt,
            fieldnorm_reader,
        ))
    }
}

impl Weight for FuzzyPhraseWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        Ok(Box::new(self.fuzzy_phrase_scorer(reader, boost)?))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.fuzzy_phrase_scorer(reader, 1.0)?;
        // A fresh scorer is positioned on its first match, which may already
        // be past `doc`; seeking backwards would violate the seek contract.
        if scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let fieldnorm_id = fieldnorm_reader.fieldnorm_id(doc);
        let match_count = scorer.match_count();
        let mut explanation = Explanation::new("FuzzyPhraseScorer", scorer.score());
        explanation.add_detail(Explanation::new("matches", match_count as Score));
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            explanation.add_detail(similarity_weight.explain(fieldnorm_id, match_count));
        }
        Ok(explanation)
    }
}
