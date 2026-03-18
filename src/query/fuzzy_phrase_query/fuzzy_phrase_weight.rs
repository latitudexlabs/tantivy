use super::FuzzyPhraseScorer;
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::bm25::Bm25Weight;
use crate::query::boolean_query::BooleanQuery;
use crate::query::explanation::does_not_match;
use crate::query::term_query::TermQuery;
use crate::query::{EmptyScorer, Explanation, Occur, Query, Scorer, Weight};
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
    ) -> crate::Result<Option<FuzzyPhraseScorer<SegmentPostings>>> {
        let similarity_weight_opt = self
            .similarity_weight_opt
            .as_ref()
            .map(|similarity_weight| similarity_weight.boost_by(boost));
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        
        // Build a boolean query with all terms as SHOULD clauses
        // This will give us documents that have at least min_match terms
        let mut subqueries = Vec::with_capacity(self.phrase_terms.len());
        for term in self.phrase_terms.iter() {
            subqueries.push((
                Occur::Should,
                Box::new(TermQuery::new(term.clone(), IndexRecordOption::WithFreqsAndPositions)) as Box<dyn Query>
            ));
        }
        let boolean_query = BooleanQuery::with_minimum_required_clauses(subqueries, self.min_match);
        
        // Get the boolean scorer - this will iterate through docs with at least min_match terms
        let enable_scoring = crate::query::EnableScoring::disabled_from_schema(&reader.schema());
        let boolean_weight = boolean_query.weight(enable_scoring)?;
        let boolean_scorer = boolean_weight.scorer(reader, boost)?;
        
        // Collect postings for each term to check positions
        let mut term_postings_list = Vec::with_capacity(self.phrase_terms.len());
        for term in self.phrase_terms.iter() {
            if let Some(postings) = reader
                .inverted_index(term.field())?
                .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
            {
                term_postings_list.push(postings);
            } else {
                // Term doesn't exist in this segment - use empty postings
                term_postings_list.push(SegmentPostings::empty());
            }
        }

        Ok(Some(FuzzyPhraseScorer::new(
            boolean_scorer,
            term_postings_list,
            self.min_match,
            similarity_weight_opt,
            fieldnorm_reader,
        )))
    }
}

impl Weight for FuzzyPhraseWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if let Some(scorer) = self.fuzzy_phrase_scorer(reader, boost)? {
            Ok(Box::new(scorer))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let scorer_opt = self.fuzzy_phrase_scorer(reader, 1.0)?;
        if scorer_opt.is_none() {
            return Err(does_not_match(doc));
        }
        let mut scorer = scorer_opt.unwrap();
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let fieldnorm_id = fieldnorm_reader.fieldnorm_id(doc);
        let match_count = scorer.match_count();
        let mut explanation = Explanation::new("FuzzyPhraseScorer", scorer.score());
        explanation.add_detail(Explanation::new(
            "matches",
            match_count as Score,
        ));
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            explanation.add_detail(similarity_weight.explain(fieldnorm_id, match_count));
        }
        Ok(explanation)
    }
}
